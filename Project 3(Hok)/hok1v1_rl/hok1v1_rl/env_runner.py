"""Environment Runner —— HoK 1v1 环境封装、动作采样与轨迹收集。

────────────────────────────────────────────────────────────────────────────
B 组问题的修复说明（原实现的所有官方 API 误用）
────────────────────────────────────────────────────────────────────────────
原实现构造环境时写的是::

    HoK1v1(mode=..., camp=Camp.BLUE, server_addr=..., server_port=...,
           ai_server_addr=..., port_begin=...)

**六个关键字参数全部不存在**，且四个必填位置参数一个都没给。
官方签名（hok_env/hok/hok1v1/env1v1.py:29-38）::

    HoK1v1(runtime_id, game_launcher, lib_processor, addrs,
           eval_mode=False, predict_frequency=3, aiserver_ip="127.0.0.1")

其它连带修正：
  * ``from hok.hok1v1.hero_config import default_mode`` —— 不存在。
    hero_config.py 全文只有 ``get_default_hero_config()``。
    阵容请用 ``hok.common.camp.camp_iterator_1v1_roundrobin_camp_heroes``。
  * ``from hok.common.camp import Camp`` —— 不存在。camp.py 只有
    ``HERO_DICT`` / ``GameMode`` / ``camp_iterator_*``，没有 Camp 枚举。
  * ``env.reset()`` 返回的是 **4 元组** ``(obs, reward, done, info)``，
    不是 state 列表；state 在 ``info`` 里，且键名是 ``observation`` 而不是 ``feature``。
  * ``env.step(actions)`` 需要 **PLAYER_NUM=2 个动作**。对手通过
    ``use_common_ai=[False, True]`` 交给内置 AI，此时只需为我方决策，
    另一方传全 0（noop）。
  * ``addrs`` 是**每个玩家一个** zmq 地址（1v1 用两个连续端口）。

参考实现：官方最小可运行示例 hok_env/hok/hok1v1/unit_test/test_env.py:71-111。

────────────────────────────────────────────────────────────────────────────
C3 问题的修复说明（172 → 84 压缩 + target 的 button 条件化）
────────────────────────────────────────────────────────────────────────────
环境给出的 legal_action 是 **172 维展开格式**（最后 96 = 12 button × 8 target）；
训练样本需要 **84 维压缩格式**（官方 SERI_VEC_SPLIT_SHAPE 的第二段）。
换算必须用**已采样的 button** 去选 target 行，这一步原实现完全缺失，
导致 target 头的 mask 与其实际 button 不匹配，会采样出该 button 下的非法目标。

本实现的两个关键点：
  1. 采样 target 时，从 (12,8) 中取当前 button 那一行作为 mask；
  2. 落盘训练样本时，用 ``compress_legal_action`` 把 172 压成 84。
这样 learner 里朴素的 ``legal_actions[..., 76:84]`` 切片就恰好是
"该动作的 button 所对应的 target mask"，语义自洽。

────────────────────────────────────────────────────────────────────────────
关于 BPTT 与 chunk
────────────────────────────────────────────────────────────────────────────
LSTM 需要时间结构，但整局上千步无法一次反传。因此把一局的决策序列按
``lstm_time_steps``（默认 16）切成 chunk，并记录每个 chunk **起始处的 LSTM 状态**；
learner 在 chunk 内做 BPTT，chunk 之间通过初始状态连接（梯度被截断）。
不足一个 chunk 的尾巴会被丢弃（相对整局长度可忽略）。
"""

from typing import Dict, List, Optional, Tuple
from itertools import cycle
import logging
import time

import numpy as np
import torch

from .action_mask import (
    LEGAL_ACTION_TOTAL_COMPRESSED,
    LEGAL_ACTION_TOTAL_EXPANDED,
    LABEL_SIZE_LIST,
    compress_legal_action,
    masked_softmax,
    probs_to_log_probs,
    sample_from_probs,
    split_legal_action_expanded,
)
from .reward import (
    DEFAULT_WEIGHTS,
    ORGAN_TYPE_CRYSTAL,
    ORGAN_TYPE_TOWER,
    REWARD_COMPONENT_NAMES,
    compute_value_targets,
    determine_outcome,
    extract_hero_snapshot,
    extract_organ_hp,
    find_organ_hp,
    state_reward_terms,
)

logger = logging.getLogger(__name__)

#: 每个玩家固定 6 维动作（button, move_x, move_z, kill_x, kill_z, target）
ACTION_DIM = len(LABEL_SIZE_LIST)

#: 交给内置 common_ai 的玩家使用全 0 动作（noop）
NOOP_ACTION: Tuple[int, ...] = tuple([0] * ACTION_DIM)


def bypass_proxy_for_localhost() -> None:
    """让本机 gamecore 的 HTTP 请求绕过系统代理。

    ⚠️ 这不是可有可无的洁癖：实测 WSL 里若设置了 ``http_proxy``
    （例如为 git/pip 配的本地代理 127.0.0.1:10090），
    ``requests``（hok_env 的 GamecoreClient 用的就是它）会**把
    http://127.0.0.1:23432/v2/newGame 也发给代理**，
    代理返回 **502 Bad Gateway**，训练在 reset 阶段就报
    ``HTTPError: 502 Server Error``，而 gamecore 侧日志里什么都看不到
    （请求根本没到它）。排查成本极高，所以在这里兜住。

    只对 127.0.0.1 / localhost 生效，其余流量仍走原代理。
    """
    import os

    bypass = "127.0.0.1,localhost,::1"
    for key in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(key, "")
        parts = [p for p in cur.split(",") if p]
        for b in bypass.split(","):
            if b not in parts:
                parts.append(b)
        os.environ[key] = ",".join(parts)

#: 官方 aiarena/1v1/actor/custom.py 的 20 个英雄 id -> one-hot 下标。
#: 官方把这一 **20 维英雄身份 one-hot 拼在观测末尾**，观测因此是 725 而不是 705。
#: 本项目每局用 camp_iterator_1v1_roundrobin_camp_heroes 轮换英雄，
#: 缺了这 20 维，同一个网络就得在"不知道自己是谁"的情况下学所有英雄的打法。
#: 注意官方只拼**我方**英雄身份（敌方英雄身份要靠观测特征自己推断）。
HERO_ID_INDEX_DICT = {
    112: 0, 121: 1, 123: 2, 131: 3, 132: 4,
    133: 5, 140: 6, 141: 7, 146: 8, 150: 9,
    154: 10, 157: 11, 163: 12, 169: 13, 175: 14,
    182: 15, 193: 16, 199: 17, 502: 18, 513: 19,
}

#: 英雄身份 one-hot 的维度（= HERO_ID_INDEX_DICT 的条目数）
HERO_IDENTITY_DIM = len(HERO_ID_INDEX_DICT)


# ---------------------------------------------------------------------------
# 环境构造
# ---------------------------------------------------------------------------

def auto_detect_ai_server_ip() -> str:
    """探测 Windows 侧 gamecore 能连回的本机 IP。

    官方 ``env1v1._get_server`` 会把 ``(aiserver_ip, zmq端口)`` 发给 gamecore，
    由 gamecore（Windows 侧）主动连回本机的 zmq server。用 UDP connect 技巧
    取"发往外部"时使用的本机 IP（不真正发包）；失败回退 127.0.0.1。

    ⚠️ 镜像网络模式下返回的是 Windows 自己的适配器 IP，Windows 连回该 IP
    会 SYN 挂起，对局无法开始。此时应显式配置 ai_server_addr: 127.0.0.1。
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def build_hok_env(cfg, runtime_id: str = "hok1v1rl-train"):
    """按官方签名构造 ``HoK1v1`` 实例。

    需要三件东西（都来自 hok_env 包）：
      * ``lib_processor``：``hok.hok1v1.lib.interface.Interface``
        （原生扩展，只在 Python 3.6–3.9 下可加载，见 config.py 顶部说明）
      * ``game_launcher``：``hok.common.gamecore_client.GamecoreClient``
      * ``addrs``：每个玩家一个 zmq 地址

    Args:
        cfg: Config
        runtime_id: 本次运行的标识，gamecore 侧用它区分实例
    Returns:
        HoK1v1 实例（尚未 reset）
    """
    # gamecore 拼出的 game_id = "kaiwu-" + runtime_id + "-..."，模拟器不允许含 '_'，
    # 否则 Windows 侧模拟器直接退出，训练会在 zmq 收帧处无限卡死且无报错，提前拦截。
    if "_" in runtime_id:
        raise ValueError(
            "runtime_id 不能包含下划线（模拟器会拒绝该 game_id）: %r" % runtime_id
        )
    # 本机 gamecore 的 HTTP 必须绕过系统代理，否则会得到 502（见函数说明）
    bypass_proxy_for_localhost()
    # 延迟 import：使本模块在未安装 hok_env 时仍可被导入（便于纯逻辑单测）
    import hok.hok1v1.lib.interface as interface
    from hok.hok1v1.env1v1 import HoK1v1, interface_default_config
    from hok.hok1v1.hero_config import get_default_hero_config
    from hok.common.gamecore_client import GamecoreClient, SimulatorType

    lib_processor = interface.Interface()
    lib_processor.Init(interface_default_config)

    # ai_server_addr 为空时自动探测（gamecore 跑在远端机器时才有意义）。
    # addrs 绑定地址与 aiserver_ip 保持一致：Windows 侧按该 IP 连回 zmq，
    # 绑定 0.0.0.0 在镜像网络模式下 Windows 无法经回环访问（SYN 挂起）。
    aiserver_ip = cfg.env.ai_server_addr or auto_detect_ai_server_ip()
    addrs = [
        f"tcp://{aiserver_ip}:{cfg.env.zmq_port_begin + i}"
        for i in range(cfg.env.player_num)
    ]

    game_launcher = GamecoreClient(
        server_addr=cfg.env.gamecore_server_addr,
        gamecore_req_timeout=cfg.env.gamecore_req_timeout,
        default_hero_config=get_default_hero_config(),
        max_frame_num=cfg.env.max_frame_num,
        simulator_type=SimulatorType.RemoteRepeat,
    )

    env = HoK1v1(
        runtime_id,
        game_launcher,
        lib_processor,
        addrs,
        predict_frequency=cfg.env.predict_frequency,
        aiserver_ip=aiserver_ip,
    )
    logger.info(
        "HoK1v1 已构造: runtime_id=%s addrs=%s gamecore=%s aiserver_ip=%s",
        runtime_id, addrs, cfg.env.gamecore_server_addr, aiserver_ip,
    )
    return env


def make_camp_config(hero_ids: Optional[List[int]] = None) -> dict:
    """生成一局 1v1 的阵容配置。

    官方通过 ``camp_iterator_1v1_roundrobin_camp_heroes`` 轮换英雄，返回形如::

        {'mode': '1v1', 'heroes': [[{'hero_id': 132}], [{'hero_id': 133}]]}

    Args:
        hero_ids: 候选英雄 id；None 表示用 HERO_DICT 的全部英雄
    Returns:
        camp_config dict
    """
    from hok.common.camp import HERO_DICT, camp_iterator_1v1_roundrobin_camp_heroes

    ids = list(hero_ids) if hero_ids else list(HERO_DICT.values())
    return next(camp_iterator_1v1_roundrobin_camp_heroes(ids))


# ---------------------------------------------------------------------------
# EnvRunner
# ---------------------------------------------------------------------------

class EnvRunner:
    """单环境 HoK 1v1 runner：负责 reset / step / 采样 / 轨迹收集。

    多环境并行需要为每个环境 fork 独立子进程（``interface.Interface`` 是
    进程内单例，且每个玩家占用固定 zmq 端口），当前只支持单环境
    （``cfg.env.num_envs == 1``）。

    Args:
        env:         已构造的 HoK1v1（尚未 reset）
        network:     HoK1v1Network
        cfg:         Config
        device:      torch.device
        player_id:   我方控制的玩家索引（对手走 common_ai，固定 0）
        camp_config: 阵容配置；None 则调用 make_camp_config()
    """

    def __init__(self, env, network, cfg, device,
                 player_id: int = 0,
                 camp_config: Optional[dict] = None):
        self.env = env
        self.network = network
        self.cfg = cfg
        self.device = device
        self.player_id = player_id
        self.camp_config = camp_config

        # camp_config 为 None 时按官方迭代器每局轮换英雄阵容
        self._camp_cycle = None
        if camp_config is None:
            from hok.common.camp import (
                HERO_DICT,
                camp_iterator_1v1_roundrobin_camp_heroes,
            )
            hero_ids = list(HERO_DICT.values())
            if getattr(cfg.network, "use_hero_identity", True):
                # 英雄身份 one-hot 只认得官方那 20 个英雄（HERO_ID_INDEX_DICT）。
                # 若从全量 HERO_DICT 轮换，遇到表外英雄（实测出现过 config_id=176）
                # one-hot 只能全零，"我是谁"这个信号就白丢了。
                # 官方 baseline 的 1v1 训练也正是用这 20 个英雄，
                # 因此这里把阵容池收敛过去，与官方保持一致。
                keep = [h for h in hero_ids if h in HERO_ID_INDEX_DICT]
                if keep:
                    if len(keep) != len(hero_ids):
                        logger.info(
                            "英雄身份 one-hot 已启用：阵容池由 %d 个英雄收敛到官方 %d 个"
                            "（其余英雄的 one-hot 只能全零，等于没有身份信息）",
                            len(hero_ids), len(keep),
                        )
                    hero_ids = keep
                else:
                    logger.warning(
                        "HERO_DICT 与官方 HERO_ID_INDEX_DICT 没有交集，"
                        "英雄身份 one-hot 将永远全零；请检查 hok_env 版本。"
                    )
            self._camp_cycle = cycle(
                camp_iterator_1v1_roundrobin_camp_heroes(hero_ids)
            )

        self.max_frame_num = cfg.env.max_frame_num
        self.T = cfg.network.lstm_time_steps

        # 英雄身份：官方在观测末尾追加 20 维 one-hot，网络输入因此是
        # feature_dim(=725)，而环境原始观测是 raw_feature_dim(=705)。
        # ⚠️ 必须在下面构造 _camp_cycle 之前定好，轮换英雄池要用到它。
        self.use_hero_identity = bool(
            getattr(cfg.network, "use_hero_identity", True)
        )
        self.raw_feature_dim = cfg.network.feature_dim - (
            HERO_IDENTITY_DIM if self.use_hero_identity else 0
        )
        self._warned_hero_identity = False

        # use_common_ai[i] == True 表示玩家 i 由内置 AI 控制，我们不给动作。
        # 我方自己控制，其余玩家交给 common_ai。
        self.use_common_ai = [
            (i != self.player_id) for i in range(cfg.env.player_num)
        ]

        self._step_count = 0

        # 状态快照与阵营：逐步差分奖励与胜负判定用。训练奖励不再使用
        # gamecore 的 reward 向量（实测其 dead 分量把小兵死亡也算进去、
        # kill 权重与仓库 config.json 不符），改为从 req_pb 的英雄/塔
        # 字段读精确状态做差分，见 _take_snapshot。
        # ⚠️ 阵营无法在构造时取：env.player_camp 要到 env.reset 之后才
        # 创建，且以 runtime_id 为键。由 _resolve_camps 在对局内从
        # req_pb 反查（旧代码在这里访问 player_camp 会直接 AttributeError，
        # 曾依赖 _terminal_outcome 的 try 静默吞掉）。
        self._camps = (None, None)
        self._prev_snapshot = None

        # 当前是否有一局游戏在运行（reset 后为 True，done 后为 False），
        # 以及运行中的对局是否已自然结束（done）。官方 env.is_gameover
        # 只在收到 gameover 帧时更新，自然结束与截断的关闭路径不同，
        # 这里自己跟踪两个状态。
        self._game_running = False
        self._game_done = False

        # sub_action_mask 取不到时的降级告警（只打一次）
        self._warned_missing_sub_action_mask = False

    # -- 观测拼装 ---------------------------------------------------------

    def build_observation(self, state: Dict) -> np.ndarray:
        """把环境 state 里的观测补齐成**网络真正吃的输入**。

        复刻官方 ``aiarena/1v1/actor/custom.py::Agent.append_hero_identity``：
        用 ``state["player_id"]``（我方英雄的 runtime_id）在
        ``req_pb.hero_list`` 里找到该英雄的 ``config_id``（英雄 id），
        转成 20 维 one-hot 后拼到观测末尾。705 -> 725。

        ⚠️ 训练与评估（``deterministic=True``）必须都走这个函数。
        少走一次就又变成"采样侧与训练侧不是同一个前向"的老问题。

        解析不到英雄 id 时（罕见）保留全零并告警一次 —— 官方也是置零。
        """
        obs = np.asarray(state["observation"], dtype=np.float32).reshape(-1)
        if not self.use_hero_identity:
            return obs
        if obs.shape[0] == self.raw_feature_dim + HERO_IDENTITY_DIM:
            # 上游已经补齐过，避免重复拼接
            return obs

        vec = np.zeros(HERO_IDENTITY_DIM, dtype=np.float32)
        hero_id = None
        req_pb = state.get("req_pb")
        my_rid = state.get("player_id")
        if req_pb is not None and my_rid is not None:
            for hero in req_pb.hero_list:
                if getattr(hero, "runtime_id", None) == my_rid:
                    hero_id = getattr(hero, "config_id", None)
                    break

        idx = None
        if hero_id is not None:
            try:
                idx = HERO_ID_INDEX_DICT.get(int(hero_id))
            except (TypeError, ValueError):
                idx = None
        if idx is not None:
            vec[idx] = 1.0
        elif not self._warned_hero_identity:
            self._warned_hero_identity = True
            logger.warning(
                "无法解析我方英雄 id（player_id=%r, config_id=%r），"
                "英雄身份 one-hot 置零；若持续出现请检查 req_pb.hero_list。",
                my_rid, hero_id,
            )

        if obs.shape[0] != self.raw_feature_dim:
            raise ValueError(
                f"环境观测维度 {obs.shape[0]} != 期望的原始维度 "
                f"{self.raw_feature_dim}（= feature_dim "
                f"{self.raw_feature_dim + HERO_IDENTITY_DIM} - 英雄 one-hot "
                f"{HERO_IDENTITY_DIM}）。请核对 network.feature_dim 与 "
                f"network.use_hero_identity 是否与观测布局一致。"
            )
        return np.concatenate([obs, vec], axis=0)

    # -- reset / step -----------------------------------------------------

    def reset(self) -> Dict:
        """重置环境并返回我方玩家的 state dict。

        ``env.reset()`` 返回 ``(obs, reward, done, info)`` 四元组，
        其中 ``info``（官方内部变量名叫 ``state``）才是含
        ``observation`` / ``legal_action`` / ``reward`` / ``done`` 的 dict 列表。
        这里直接解包取出我方那一份。
        """
        # 上一局若还在运行，先关停并等它消失（见 _close_previous_game）。
        self._close_previous_game()

        # 每局重建 zmq server：旧 socket 上可能残留已死模拟器的连接路由
        # 与缓冲帧，复用会导致新对局的响应发到死连接被丢弃、对局停摆、
        # server 重启游戏后新旧 sgame_id 串流（GET_RESP_FAILED）。
        # Delete 之后 env.reset() 会重新 Add（官方 retry 路径就是这么用的）。
        for addr in self.env.addrs:
            try:
                self.env.lib_processor.server_manager.Delete(addr)
            except Exception:      # noqa: BLE001 - 删除失败就让 env.reset 处理
                logger.debug("删除 zmq server %s 失败（忽略）", addr)

        if self._camp_cycle is not None:
            camp_config = next(self._camp_cycle)
        else:
            camp_config = self.camp_config

        _obs, _reward, _done, states = self.env.reset(
            camp_config,
            use_common_ai=self.use_common_ai,
            eval=False,
        )
        self._step_count = 0
        self._camps = (None, None)
        self._prev_snapshot = None
        self._game_running = True
        self._game_done = False
        return states[self.player_id]

    def step(self, action6):
        """推进环境一个决策点。

        奖励由 req_pb 状态差分计算（compute_state_reward），不使用
        gamecore 的 reward 向量（该向量语义实测混乱，见 reward.py
        模块 docstring）。分量 dict 随返回值带出供逐局统计使用。

        Args:
            action6: 长度 6 的动作（仅我方）
        Returns:
            (state_dict, reward_scalar, done_bool, components_dict)
        """
        # 每个玩家都要给动作；对手交给 common_ai，传 noop 即可
        actions = [NOOP_ACTION] * self.cfg.env.player_num
        actions[self.player_id] = tuple(int(x) for x in action6)

        _obs, reward, done, states = self.env.step(actions)
        self._step_count += 1

        snap = self._take_snapshot(states[self.player_id])
        if snap is None:
            # req_pb 异常（罕见帧）：跳过本步差分，只给时间惩罚。
            # ⚠️ 原实现这里又赋了一次 -time_coef，而下面还会再 += 一次，
            #    使异常帧实际承担了双倍时间惩罚；统一改为 0.0。
            components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            reward_scalar = 0.0
        elif self._prev_snapshot is None:
            # 第一步没有差分基准：只给时间惩罚
            self._prev_snapshot = snap
            components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            reward_scalar = 0.0
        else:
            # 单一真源：与 reward.compute_state_reward 共用同一套公式，
            # 且各分量已是**加权后**的贡献，因此这里恒有
            #     sum(components.values()) == reward_scalar
            # 日志/看板看到的"哪个分量在主导奖励"因此可以直接解释回报。
            components = state_reward_terms(self._prev_snapshot, snap)
            reward_scalar = float(sum(components.values()))
            self._prev_snapshot = snap

        # 时间惩罚：每步常数 -coef，逼迫尽快终结比赛。
        time_penalty = -self.cfg.env.reward_time_coef
        reward_scalar += time_penalty
        components["time"] = time_penalty

        done_bool = bool(done[self.player_id])
        if done_bool:
            self._game_running = False
            self._game_done = True
        return (
            states[self.player_id],
            float(reward_scalar),
            done_bool,
            components,
        )

    def _resolve_camps(self, req_pb):
        """从 req_pb 反查我方/敌方阵营。

        env.player_list[self.player_id] 是我方英雄的 runtime_id（对局开始
        后由 gamecore 填充），在 hero_list 里找到该英雄的 camp 即我方阵营，
        另一个英雄就是敌方。对局未开始（runtime_id 还是 None）或英雄缺席
        时返回 (None, None)。
        """
        my_rid = self.env.player_list[self.player_id]
        if my_rid is None:
            return None, None
        my_camp = None
        for hero in req_pb.hero_list:
            if getattr(hero, "runtime_id", None) == my_rid:
                my_camp = hero.camp
                break
        if my_camp is None:
            return None, None
        for hero in req_pb.hero_list:
            if hero.camp != my_camp:
                return my_camp, hero.camp
        return None, None

    def _take_snapshot(self, state):
        """读取 req_pb 中的英雄/塔状态快照。

        返回 dict（见 compute_state_reward 的 prev/cur 结构）；req_pb
        缺失、英雄缺席或阵营未解析（罕见帧）时返回 None，调用方跳过本步差分。

        ⚠️ ``observer_camp=my_camp`` 是必需的：它让快照带上"敌人对我是否可见"，
        奖励侧才能避开"不可见时 hp 是占位值(1)"这个坑（见 reward.hp_term）。
        """
        req_pb = state.get("req_pb")
        if req_pb is None:
            return None
        my_camp, en_camp = self._camps
        if my_camp is None:
            my_camp, en_camp = self._resolve_camps(req_pb)
            if my_camp is None:
                return None
            self._camps = (my_camp, en_camp)
        my = extract_hero_snapshot(req_pb, my_camp, observer_camp=my_camp)
        en = extract_hero_snapshot(req_pb, en_camp, observer_camp=my_camp)
        if my is None or en is None:
            return None
        my_org = sum(extract_organ_hp(req_pb, my_camp))
        en_org = sum(extract_organ_hp(req_pb, en_camp))
        return {"my": my, "en": en, "my_organ": my_org, "en_organ": en_org}

    def _close_previous_game(self):
        """关停上一局并等 server 侧任务消失，再开下一局。

        不依赖官方 ``env.close_game()`` 排空尾帧（游戏停摆时它会无限刷日志
        卡死）：python 侧 zmq 缓冲问题由 reset 里的重建 server 解决，
        这里只负责把 server 侧的对局关掉。
        """
        if not self._game_running:
            return
        self._stop_game_quietly()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                if not self.env.game_launcher.check_exists_game(self.env.runtime_id):
                    break
            except Exception:      # noqa: BLE001 - 轮询失败就继续等
                pass
            time.sleep(0.5)
        time.sleep(0.5)            # zmq 连接拆除的缓冲时间
        self._game_running = False

    def _stop_game_quietly(self):
        """通过 gamecore-server 强制关停当前对局（HTTP，秒回）。

        用于截断 / 放弃的对局：官方 ``env.close_game()`` 会
        ``while not is_gameover`` 等游戏自然结束，停摆游戏会无限刷日志卡死
        （实测 8 分钟刷出 4.4GB 日志）。stop_game 由 gamecore-server 侧杀掉
        对局，zmq 连接随之断开，下一局从干净状态开始。
        """
        try:
            self.env.game_launcher.stop_game(self.env.runtime_id)
        except Exception as exc:      # noqa: BLE001 - 关闭失败不应中断训练
            logger.warning("stop_game 失败（可忽略）：%s", exc)

    def close(self):
        """停止当前对局并释放资源（若还有对局在跑）。"""
        self._close_previous_game()

    # -- 采样 -------------------------------------------------------------

    @torch.no_grad()
    def sample_action(self, feature: np.ndarray, legal_action_172: np.ndarray,
                      lstm_state, deterministic: bool = False,
                      random_policy: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, Tuple[torch.Tensor, torch.Tensor], float]:
        """对单帧观测做 6 头分层 masked 采样。

        流程：
          1. 网络前向：head 0-4 输出 (1,1,size)，head 5 输出 (1,1,8)
          2. head 0-4：用 172 维展开 mask 的对应片段做 masked_softmax 后采样
          3. 用**已采样的 button** 从 (12,8) 里取出该 button 的 8 维 target mask
          4. head 5：用上一步的 mask 做 masked_softmax 后采样
          5. 调用方随后用 button 把 172 压成 84，作为训练样本

        Args:
            feature:          (725,) 观测
            legal_action_172: (172,) 展开 legal_action
            lstm_state:       (h, c)，各 (1, 1, hidden)
            deterministic:    True 时取 argmax（评估用）
            random_policy:    True 时**忽略网络输出**、在每个头的合法动作里
                均匀随机采样。仅用于诊断探针（对照实验需要多样化对局），
                训练路径不使用。此时仍会前向以更新 lstm_state。
        Returns:
            (action6 (6,) int64, log_probs6 (6,) float32, lstm_state_new, value)
        """
        self.network.eval()

        feat = torch.as_tensor(feature, dtype=torch.float32, device=self.device)
        feat = feat.reshape(1, 1, -1)                      # (B=1, T=1, 725)

        out = self.network(feat, lstm_state)
        logits_list = out["logits_list"]
        value = float(out["value"].reshape(-1)[0].item())
        lstm_state_new = out["lstm_state_new"]

        # 172 维展开 mask 切分 -> 前 5 个一维 mask + 一个 (12,8) 的 target mask
        la = torch.as_tensor(legal_action_172, dtype=torch.float32, device=self.device)
        splits = split_legal_action_expanded(la)

        actions: List[int] = []
        log_probs: List[float] = []

        # ---- head 0-4：各自独立的 mask ----
        for i in range(5):
            logits_i = logits_list[i].reshape(1, -1)        # (1, size_i)
            mask_i = splits[i].reshape(1, -1)               # (1, size_i)
            probs_i = masked_softmax(logits_i, mask_i)
            if random_policy:
                # ⚠️ 形状必须与 sample_from_probs 的返回值**完全一致**：
                #    它是 torch.multinomial(flat,1).squeeze(-1).reshape(probs.shape[:-1])，
                #    即保留 batch 维 -> 这里 probs_i 是 (1, size_i)，故 a_i 必须是 (1,)。
                #    写成 0 维会让下面 a_i.unsqueeze(-1) 变成 (1,)，
                #    与 probs_i 的 (1, size_i) 秩不匹配（gather 报错）。
                legal_idx = torch.nonzero(
                    mask_i.reshape(-1) > 0.5, as_tuple=False).reshape(-1)
                if legal_idx.numel() == 0:
                    a_i = torch.zeros((1,), dtype=torch.long, device=logits_i.device)
                else:
                    k = int(torch.randint(legal_idx.numel(), (1,)).item())
                    a_i = legal_idx[k:k + 1]
            else:
                a_i = sample_from_probs(probs_i, deterministic=deterministic)
            actions.append(int(a_i.item()))
            p = probs_i.gather(-1, a_i.unsqueeze(-1)).squeeze(-1)
            # log 概率用 probs_to_log_probs（内部 +1e-5，与官方 MIN_POLICY 一致）
            log_probs.append(float(probs_to_log_probs(p).item()))

        # ---- head 5 (target)：mask 由已采样 button 决定（C3 的关键）----
        button = actions[0]
        target_mask_2d = splits[5]                          # (12, 8)
        target_mask = target_mask_2d[button]                # (8,)  第 button 行的合法 target

        logits_t = logits_list[5].reshape(1, -1)            # (1, 8)
        probs_t = masked_softmax(logits_t, target_mask.reshape(1, -1))
        if random_policy:
            legal_t = torch.nonzero(
                target_mask.reshape(-1) > 0.5, as_tuple=False).reshape(-1)
            if legal_t.numel() == 0:
                a_t = torch.zeros((1,), dtype=torch.long, device=logits_t.device)
            else:
                k = int(torch.randint(legal_t.numel(), (1,)).item())
                a_t = legal_t[k:k + 1]
        else:
            a_t = sample_from_probs(probs_t, deterministic=deterministic)
        actions.append(int(a_t.item()))
        p_t = probs_t.gather(-1, a_t.unsqueeze(-1)).squeeze(-1)
        log_probs.append(float(probs_to_log_probs(p_t).item()))

        action6 = np.asarray(actions, dtype=np.int64)
        logp6 = np.asarray(log_probs, dtype=np.float32)
        return action6, logp6, lstm_state_new, value

    # -- 轨迹收集 ---------------------------------------------------------

    def collect_trajectory(self, max_decisions: Optional[int] = None,
                           deterministic: bool = False,
                           random_policy: bool = False
                           ) -> Dict[str, object]:
        """跑完一整局，收集 PPO 更新所需的样本。

        输出按 ``lstm_time_steps`` 切成 chunk（见模块 docstring 的 BPTT 说明），
        每个 chunk 记录其起始 LSTM 状态。

        Args:
            max_decisions: 单局最大决策数；默认为
                ``max_frame_num // predict_frequency``（帧数换算成决策点）
            deterministic: True 时动作取 argmax（评估对局用，不打入训练 batch）
            random_policy: True 时动作在合法集里均匀随机（**仅诊断探针用**，
                例如跑对照实验统计胜负判定；训练路径不要开）
        Returns:
            dict:
              features (B,T,725), legal_actions (B,T,84), actions (B,T,6),
              log_probs (B,T,6), values (B,T), rewards (B,T), dones (B,T),
              advantages (B,T), returns (B,T), head_weights (B,T,6),
              lstm_state_init ((1,B,H), (1,B,H)),
              episode_len (int), win (bool),
              episode_stats (dict): 奖励分量求和、6 个动作头计数、胜负，
                供训练日志与可视化看板使用
        """
        if max_decisions is None:
            # +8 余量：游戏在 max_frame_num 帧处自行结束（gameover 帧会作为
            # 最终 AI 帧送达）。若按 max_frame_num // predict_frequency 恰好
            # 在 gameover 前一步截断，游戏永远处于"未结束"状态，下一局 reset
            # 时会串进旧对局的尾帧（GET_RESP_FAILED），必须让对局自然结束。
            max_decisions = max(
                1, self.max_frame_num // max(1, self.cfg.env.predict_frequency)
            ) + 8
        T = self.T

        # 逐决策缓冲
        feats: List[np.ndarray] = []
        las84: List[np.ndarray] = []
        acts: List[np.ndarray] = []
        lps: List[np.ndarray] = []
        vals: List[float] = []
        rews: List[float] = []
        dones: List[float] = []
        head_w: List[np.ndarray] = []
        chunk_states: List[Tuple[np.ndarray, np.ndarray]] = []

        state = self.reset()
        lstm_state = self.network.init_hidden(1, self.device)
        win = False
        decisions_in_chunk = 0
        truncated = True
        #: 终局判定的原始证据；只有自然结束时才由 _outcome_evidence 填充
        #: （截断不产生胜负证据，保持 None 以免被误当成"判定依据"）。
        outcome_evidence = None

        # 逐局统计：奖励分量求和 + 6 头动作计数（看板数据源）
        comp_sum = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        comp_sum["extra"] = 0.0
        comp_sum["time"] = 0.0     # 时间惩罚（step 里追加的分量）
        comp_sum["truncate"] = 0.0  # 截断惩罚（达到 max_decisions 时非零）
        action_counts = [
            np.zeros(size, dtype=np.int64) for size in LABEL_SIZE_LIST
        ]

        for _ in range(max_decisions):
            feature = self.build_observation(state)
            la172 = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)

            if la172.shape[0] != LEGAL_ACTION_TOTAL_EXPANDED:
                raise ValueError(
                    f"legal_action 维度 {la172.shape[0]} != {LEGAL_ACTION_TOTAL_EXPANDED}；"
                    "环境原始输出应为 172 维展开格式，"
                    "请确认没有把 84 维压缩格式误当成环境输出"
                )

            # chunk 边界：记录本 chunk 的初始 LSTM 状态
            # （detach + cpu，断开跨 chunk 的梯度，实现截断 BPTT）
            if decisions_in_chunk == 0:
                chunk_states.append((
                    lstm_state[0].detach().clone().cpu().numpy(),
                    lstm_state[1].detach().clone().cpu().numpy(),
                ))

            action6, logp6, lstm_state, value = self.sample_action(
                feature, la172, lstm_state, deterministic=deterministic,
                random_policy=random_policy,
            )

            # 官方的 weight0..5 = sub_action_mask[选中的 button]：一个 6 维 0/1 向量，
            # 说明"选了该 button 之后，这 6 个动作头里哪些真正生效"。
            # learner 用它屏蔽与本 button 无关的头（否则这些头会白白进入 ratio
            # 与熵，把 KL 抬高、触发早停）。环境不提供时退化为全 1。
            sub_mask = state.get("sub_action_mask")
            w6 = None
            if isinstance(sub_mask, dict):
                btn = int(action6[0])
                # 键可能是 int 也可能是 str，两种都试
                for key in (btn, str(btn)):
                    if key in sub_mask:
                        cand = np.asarray(
                            sub_mask[key], dtype=np.float32
                        ).reshape(-1)
                        if cand.shape[0] == ACTION_DIM:
                            w6 = cand
                        break
            if w6 is None:
                # 取不到就退化为不加权。注意这会**静默**失去官方的 head 屏蔽，
                # 所以第一次退化时明确告警，避免又被当成"调参问题"。
                if not self._warned_missing_sub_action_mask:
                    self._warned_missing_sub_action_mask = True
                    logger.warning(
                        "state 中缺少可用的 sub_action_mask，本次训练不做 head 屏蔽"
                        "（官方的 weight0..5 缺失）。请检查 hok_env 版本。"
                    )
                w6 = np.ones(ACTION_DIM, dtype=np.float32)
            head_w.append(w6)

            # 172 -> 84：用本次采样的 button 选 target 行（C3 的关键）
            la84 = compress_legal_action(
                torch.as_tensor(la172, dtype=torch.float32),
                torch.as_tensor(action6[0], dtype=torch.long),
            ).numpy().astype(np.float32)
            assert la84.shape[0] == LEGAL_ACTION_TOTAL_COMPRESSED

            # 记录 t 时刻的一条转移。奖励先占位，等 step 返回后回填到
            # **同一个下标**上：rews[t] 必须与 (s_t, a_t, V_t) 对齐。
            # ⚠️ 原实现是在 step **之前** append 上一次的 reward，
            #    使整条轨迹的奖励相对状态/动作滞后一步
            #    （rews[t] 实际是 a_{t-1} 的回报），GAE 的
            #    delta_t = r_t + γV_{t+1} - V_t 因此把信用记到了错的动作上。
            feats.append(feature)
            las84.append(la84)
            acts.append(action6)
            lps.append(logp6)
            vals.append(value)
            rews.append(0.0)
            dones.append(0.0)

            decisions_in_chunk = (decisions_in_chunk + 1) % T

            # 推进环境（reward 是 9 维分量向量，step 内部已解析成标量）
            try:
                state, step_reward, done, components = self.step(action6)
            except Exception:
                logger.error(
                    "step 失败：本局第 %d 个决策点（约第 %d 帧）。"
                    "_game_running=%s _game_done=%s "
                    "cur_sgame_ids=%s cur_frame_no=%s is_gameover=%s",
                    len(feats),
                    len(feats) * max(1, self.cfg.env.predict_frequency),
                    self._game_running,
                    self._game_done,
                    getattr(self.env, "cur_sgame_ids", None),
                    getattr(self.env, "cur_frame_no", None),
                    getattr(self.env, "is_gameover", None),
                )
                raise
            rews[-1] = float(step_reward)      # 回填到本步（见上）
            for k, v in components.items():
                comp_sum[k] += float(v)
            for head_i, a in enumerate(action6):
                action_counts[head_i][int(a)] += 1

            if done:
                # 终局：终端奖励叠加在本步记录上（rews[-1] 就是本步奖励），
                # 并打上终止标记（GAE 会在此切断 bootstrap）
                outcome = self._terminal_outcome(state)
                win = (outcome == "win")
                #: 判定时刻的原始证据（供离线审计 / 对照实验使用）。
                #: 不改训练逻辑，只是把"凭什么这么判"记下来。
                outcome_evidence = self._outcome_evidence(state, outcome)
                if outcome is not None:
                    bonus = (
                        DEFAULT_WEIGHTS.win if outcome == "win"
                        else DEFAULT_WEIGHTS.lose
                    )
                    rews[-1] += bonus
                dones[-1] = 1.0
                truncated = False
                break

        # ================= 轨迹切分：尾部补齐，不丢终局步 =================
        # 顺序固定：截断处理 → GAE(整条轨迹) → 切 chunk 并补齐 → valid_mask
        #
        # 【为什么不能像旧代码那样丢弃尾巴】原实现 `used = num_chunks * T` 直接截断，
        #   而 n % T 几乎不可能为 0 ⇒ **每一局都丢掉含 done=1 与 ±50 终端奖励的
        #   终局步**。它长期静默，是因为 dones 数组里永远是 0，没人能看出来。
        #   官方 actor.py::_save_last_sample 会显式保存最后一个决策并强制
        #   next_value=0 —— "终局样本一定进 batch"是官方写死的保证。
        #
        # 【现在的做法】尾部不足 T 的部分用 0 补齐成完整 chunk，所有 chunk 共享同一 T
        #   （因此可被 collect_batch 正常拼接），并给出 valid_mask。
        #
        # ⚠️ 不变量：learner 必须按 valid_mask 归一化损失，否则补齐的 0
        #    （0 奖励 / 0 advantage / 0 logprob）会稀释真实样本的梯度。
        n = len(feats)
        if n == 0:
            raise RuntimeError("collect_trajectory 未收集到任何决策，请检查环境是否正常")

        if truncated:
            # 达到 max_decisions 仍未结束：按截断处理，切断 bootstrap。
            # 严格来说截断时应继续 bootstrap（状态并非真正终止态），
            # 这里与官方 baseline 一致统一按切断处理；如需区分可另加 truncated 数组。
            dones[-1] = 1.0
            # ⚠️ 截断必须给惩罚，否则"不结束对局"是严格更优的策略：
            # money/exp/tower 等 dense 项每步都在累加，而赢下对局会
            # 立刻终止这条收益流。历史症状正是 episode_len 顶到上限、
            # win_rate 长期不动（截断局既不计胜也不计负）。
            trunc_penalty = float(DEFAULT_WEIGHTS.truncate)
            if trunc_penalty:
                rews[-1] += trunc_penalty
                comp_sum["truncate"] += trunc_penalty
            logger.debug(
                "collect_trajectory 达到 max_decisions=%d，按截断处理（截断惩罚 %.2f）",
                max_decisions, trunc_penalty,
            )

        # split 必须先于 padding 完成：pad 只在**最后一个** chunk 上。
        num_chunks = max(1, (n + T - 1) // T)
        n_valid = n
        pad = num_chunks * T - n
        T_eff = T

        shape = (num_chunks, T_eff)

        def _pad(arrs, dtype=np.float32):
            """把按决策排列的 list 切+补齐成 (num_chunks, T_eff) 或 (num_chunks, T_eff, w)。

            ⚠️ 一维数组（rewards / values / advantages / returns / dones）
            必须保持 **(num_chunks, T_eff)**，不能多出一个尾维 (…, 1)：
            否则 learner 里 `ratio_i (B,T) * advantages (B,T,1)` 会广播成
            (B,B,T)，报 "tensor a (512) vs b (16)" 这种看不出根因的错误。
            """
            orig = np.asarray(arrs, dtype=dtype)
            if orig.ndim <= 1:
                arr = orig.reshape(n, 1)
                if pad:
                    arr = np.concatenate(
                        [arr, np.zeros((pad, 1), dtype=dtype)], axis=0)
                return arr.reshape(num_chunks, T_eff)
            width = orig.shape[-1]
            arr = orig.reshape(n, width)
            if pad:
                arr = np.concatenate(
                    [arr, np.zeros((pad, width), dtype=dtype)], axis=0)
            return arr.reshape(num_chunks, T_eff, width)

        # ---- GAE：必须在**整条轨迹**上先算完，再切 chunk ----
        # ⚠️ 顺序不可颠倒：advantages/returns 由下面这行产生，随后才被 _pad 打包。
        #    若把打包写在它前面，会直接 UnboundLocalError（py_compile 查不出来）。
        advantages, returns = compute_value_targets(
            rewards=np.asarray(rews, dtype=np.float32),
            gamma=self.cfg.ppo.gamma,
            values=np.asarray(vals, dtype=np.float32),
            last_value=0.0,
            lamda=self.cfg.ppo.lamda,
            dones=np.asarray(dones, dtype=np.float32),
        )

        # ---- 打包成 (chunk, T[, w]) ----
        feats_arr = _pad(feats)
        las_arr = _pad(las84)
        acts_arr = _pad(acts)
        lps_arr = _pad(lps)
        hw_arr = _pad(head_w)
        vals_arr = _pad(vals)
        rews_arr = _pad(rews)
        dones_arr = _pad(dones)
        adv_arr = _pad(advantages)
        ret_arr = _pad(returns)

        #: 1 = 真实决策，0 = 尾部补齐。learner 用它把补齐位从 loss 的归一化
        #: 分母里剔除（见 compute_loss 的 valid_mask 说明）。
        valid_arr = np.ones((n,), dtype=np.float32)
        if pad:
            valid_arr = np.concatenate(
                [valid_arr, np.zeros((pad,), dtype=np.float32)])
        valid_mask = valid_arr.reshape(shape)

        # 每个 chunk 的初始 LSTM 状态，拼成 (1, B, H)
        h0 = np.concatenate([s[0] for s in chunk_states[:num_chunks]], axis=1)
        c0 = np.concatenate([s[1] for s in chunk_states[:num_chunks]], axis=1)

        # 终局步在切分后的坐标（供 episode_stats 里的自检字段使用）
        last_step_flat = n - 1
        last_chunk_idx = last_step_flat // T_eff
        last_step_in_chunk = last_step_flat % T_eff

        return {
            "features": feats_arr,
            "legal_actions": las_arr,
            "actions": acts_arr,
            "log_probs": lps_arr,
            "values": vals_arr,
            "rewards": rews_arr,
            "dones": dones_arr,
            "advantages": adv_arr,
            "returns": ret_arr,
            "head_weights": hw_arr,
            # ⚠️ valid_mask：尾部补齐出的位置为 0，learner **必须**按它归一化损失，
            # 否则补齐的 0（0 奖励、0 advantage、0 logprob）会稀释梯度。
            "valid_mask": valid_mask,
            #: 一个 chunk 的时间长度（现在是恒定的 T；保留此键是为了让
            #: train_ppo 的拼接逻辑不必猜测时间维含义）
            "chunk_T": int(T_eff),
            "lstm_state_init": (
                torch.as_tensor(h0, dtype=torch.float32, device=self.device),
                torch.as_tensor(c0, dtype=torch.float32, device=self.device),
            ),
            "episode_len": n,
            "win": win,
            "episode_stats": {
                "win": win,
                "episode_len": n,
                "truncated": bool(truncated),
                "outcome_evidence": outcome_evidence,
                "reward_components": comp_sum,
                # ---- 终局可见性自检（P3 的哨兵）----
                # 旧实现丢弃尾部时这些值会暴露出来：padded_steps > 0 说明
                # 尾部被补齐（而不是被丢弃）；last_step_valid 必须恒为 True，
                # 否则说明终局步又掉了。
                "chunk_padded_steps": int(pad),
                "chunk_valid_ratio": float(n_valid) / float(num_chunks * T_eff),
                "last_step_chunk": int(last_chunk_idx),
                "last_step_in_chunk": int(last_step_in_chunk),
                "last_step_valid": bool(valid_mask[last_chunk_idx, last_step_in_chunk]),
                "terminal_in_batch": bool(dones_arr[last_chunk_idx, last_step_in_chunk] > 0.5),
                "action_hist": {
                    f"head_{i}": action_counts[i].tolist()
                    for i in range(ACTION_DIM)
                },
            },
        }

    def _outcome_evidence(self, state: Dict, outcome) -> Dict:
        """收集"胜负是怎么判出来的"原始证据（只读，供审计/对照实验）。

        动机：历史上 `env/win_rate` 曾与 `env/reward/tower` **互相矛盾**
        （旧日志里出现过"赢"与"丢基地"同时成立），而当时没有任何字段能
        事后审计。这里把判定依据（阵营映射、双方水晶/塔血量、建筑清单）
        原样记进 `episode_stats`，使"判定错误"与"奖励算错"可以区分开。
        """
        ev = {
            "outcome": outcome,
            "my_camp": None,
            "en_camp": None,
            "player_id": int(self.player_id),
            "my_runtime_id": None,
            "hero_camps": [],
            "organ_count": 0,
            "crystal_my": None,
            "crystal_en": None,
            "tower_my": None,
            "tower_en": None,
            "organ_total_my": None,
            "organ_total_en": None,
            "crystal_zero_camps": [],
            "tower_zero_camps": [],
        }
        try:
            req_pb = state.get("req_pb")
            if req_pb is None:
                return ev
            my_camp, en_camp = self._camps
            if my_camp is None:
                my_camp, en_camp = self._resolve_camps(req_pb)
            ev["my_camp"] = None if my_camp is None else int(my_camp)
            ev["en_camp"] = None if en_camp is None else int(en_camp)
            try:
                rid = self.env.player_list[self.player_id]
                ev["my_runtime_id"] = None if rid is None else int(rid)
            except Exception:      # noqa: BLE001
                ev["my_runtime_id"] = None
            for h in req_pb.hero_list:
                ev["hero_camps"].append({
                    "camp": int(h.camp),
                    "runtime_id": int(getattr(h, "runtime_id", -1)),
                    "hp": float(h.hp),
                    "kills": int(getattr(h, "killCnt", 0)),
                    "deaths": int(getattr(h, "deadCnt", 0)),
                })
            organs = list(req_pb.organ_list)
            ev["organ_count"] = len(organs)
            zeros = {}
            for o in organs:
                t = int(getattr(o, "type", -1))
                c = int(getattr(o, "camp", -1))
                hp = float(o.hp)
                if hp <= 0.0:
                    zeros.setdefault(t, []).append(c)
            ev["crystal_zero_camps"] = sorted(zeros.get(ORGAN_TYPE_CRYSTAL, []))
            ev["tower_zero_camps"] = sorted(zeros.get(ORGAN_TYPE_TOWER, []))
            if my_camp is not None and en_camp is not None:
                ev["crystal_my"] = find_organ_hp(req_pb, my_camp, ORGAN_TYPE_CRYSTAL)
                ev["crystal_en"] = find_organ_hp(req_pb, en_camp, ORGAN_TYPE_CRYSTAL)
                ev["tower_my"] = find_organ_hp(req_pb, my_camp, ORGAN_TYPE_TOWER)
                ev["tower_en"] = find_organ_hp(req_pb, en_camp, ORGAN_TYPE_TOWER)
                ev["organ_total_my"] = float(sum(extract_organ_hp(req_pb, my_camp)))
                ev["organ_total_en"] = float(sum(extract_organ_hp(req_pb, en_camp)))
        except Exception as exc:      # noqa: BLE001 - 取证失败不应影响训练
            logger.debug("_outcome_evidence 取证失败（忽略）：%s", exc)
        return ev

    # -- 终局判定 ---------------------------------------------------------

    def _terminal_outcome(self, state: Dict) -> Optional[str]:
        """对局结束时的胜负判定：以塔/水晶血量为准。

        实测（probe_reward.py）：对局结束时被推方的塔与水晶 hp 均为 0，
        双方英雄往往都活着。官方 actor.py 的"我方英雄 hp > 0 即 win"会把
        被推塔的对局恒定误判为 win（历史日志 win_rate 恒为 1.0 的来源），
        ±10 终端奖励因此一直在奖励输局。改用 determine_outcome：
        我方水晶归零 => lose，敌方水晶归零 => win，其余比较塔+水晶总量。
        """
        try:
            req_pb = state.get("req_pb")
            if req_pb is None:
                return None
            my_camp, en_camp = self._camps
            if my_camp is None:
                my_camp, en_camp = self._resolve_camps(req_pb)
                if my_camp is None:
                    return None
            outcome = determine_outcome(req_pb, my_camp, en_camp)
            if outcome is not None:
                return outcome
        except Exception as exc:      # noqa: BLE001
            logger.debug("terminal_outcome 判定失败，忽略终端奖励：%s", exc)
        return None
