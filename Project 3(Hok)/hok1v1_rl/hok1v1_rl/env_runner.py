"""gamecore 连接、生命周期和单步采样。整局缓冲见 rollout.py。

native Interface 为进程内单例，因此同一进程按顺序训练/评估并复用连接。
"""
from typing import Dict, Optional
from datetime import datetime
import logging
import time
import numpy as np
import torch
from .action_mask import (LABEL_SIZE_LIST, split_legal_action_expanded,
                          masked_softmax, probs_to_log_probs, sample_from_probs)
from .config import HERO_IDS
from .matchups import TrainingSchedule
from .network import configure_float32
from .reward import (REWARD_COMPONENT_NAMES, extract_hero_snapshot,
                     organ_snapshot, state_reward_terms, determine_outcome, sdk_last_hit)
logger = logging.getLogger(__name__)
ACTION_DIM = 6
NOOP_ACTION = (0, 0, 0, 0, 0, 0)
HERO_ID_INDEX_DICT = {hero: index for index, hero in enumerate(HERO_IDS)}
HERO_IDENTITY_DIM = len(HERO_IDS)
NATURAL_EXIT_TIMEOUT = 30.0
STOP_EXIT_TIMEOUT = 10.0

def state_game_ids(state):
    """SDK 返回的是客户端/协议 ID；bytes 先解码，不能冒充 ABS 文件名中的 game_id。"""
    def text(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) if value is not None else ""
    return {"sdk_game_id": text(state.get("game_id")),
            "sdk_client_id": text(state.get("sgame_id") or state.get("game_id"))}

def bypass_proxy_for_localhost() -> None:
    import os
    bypass = '127.0.0.1,localhost,::1'
    for key in ('no_proxy', 'NO_PROXY'):
        cur = os.environ.get(key, '')
        parts = [p for p in cur.split(',') if p]
        for b in bypass.split(','):
            if b not in parts:
                parts.append(b)
        os.environ[key] = ','.join(parts)

def auto_detect_ai_server_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return '127.0.0.1'

def build_hok_env(cfg, runtime_id: str='hok1v1rl-train'):
    if '_' in runtime_id:
        raise ValueError('runtime_id 不能包含下划线（模拟器会拒绝该 game_id）: %r' % runtime_id)
    bypass_proxy_for_localhost()
    import hok.hok1v1.lib.interface as interface
    from hok.hok1v1.env1v1 import HoK1v1, interface_default_config
    from hok.hok1v1.hero_config import get_default_hero_config
    from hok.common.gamecore_client import GamecoreClient, SimulatorType
    from pathlib import Path
    if not Path("config.json").is_file():
        raise FileNotFoundError("native 扩展需要工作目录中的 config.json；请 cd 到 hok1v1_rl")
    lib_processor = interface.Interface()
    lib_processor.Init(interface_default_config)
    aiserver_ip = cfg.env.ai_server_addr or auto_detect_ai_server_ip()
    addrs = [f'tcp://{aiserver_ip}:{cfg.env.zmq_port_begin + i}' for i in range(cfg.env.player_num)]
    game_launcher = GamecoreClient(
        server_addr=cfg.env.gamecore_server_addr,
        gamecore_req_timeout=cfg.env.gamecore_req_timeout,
        default_hero_config=get_default_hero_config(),
        max_frame_num=cfg.env.max_frame_num,
        # 不从历史 ABS 的中途状态开始，避免训练阵容和回放来源混在一起。
        simulator_type=SimulatorType.Remote,
    )
    env = HoK1v1(
        runtime_id, game_launcher, lib_processor, addrs,
        predict_frequency=cfg.env.predict_frequency, aiserver_ip=aiserver_ip,
    )
    logger.info('HoK1v1 已构造: runtime_id=%s addrs=%s gamecore=%s aiserver_ip=%s', runtime_id, addrs, cfg.env.gamecore_server_addr, aiserver_ip)
    return env

class EnvRunner:
    """对手为 common AI；player_id 随阵容在 0/1 间切换。"""
    def __init__(self, env, network, cfg, device, player_id=0, camp_config=None):
        self.env, self.network, self.cfg, self.device = env, network, cfg, device
        configure_float32(device)
        self.player_id, self.camp_config = player_id, camp_config
        self.schedule = TrainingSchedule(cfg.env.hero_ids, cfg.env.opponent_hero_ids, cfg.train.seed)
        self.current_matchup = None
        self.max_frame_num = cfg.env.max_frame_num
        self.T = cfg.network.lstm_time_steps
        self._camps = (None, None)
        self._prev_snapshot = None
        self._game_running = self._game_done = False
        self._step_count = 0
        self.episode_info = {}
        self.control = None
        self.inference = network
        self.reset_timings = {}
        self.progress_callback = None

    def prepare_inference(self, mode="eager"):
        """在权重同步后调用一次；逐步采样时不再遍历网络切换 eval 模式。"""
        self.network.eval()
        if mode == "jit":
            from .inference import CPUInference
            self.inference = CPUInference(self.network)
        elif mode == "eager":
            self.inference = self.network
        else:
            raise ValueError("推理模式必须为 eager 或 jit")

    def _take_snapshot(self, state):
        req_pb = state.get("req_pb")
        if req_pb is None:
            return None
        my_camp, en_camp = self._camps
        if my_camp is None:
            my_camp, en_camp = self._resolve_camps(req_pb)
            self._camps = my_camp, en_camp
        if my_camp is None:
            return None
        my = extract_hero_snapshot(req_pb, my_camp, my_camp)
        enemy = extract_hero_snapshot(req_pb, en_camp, my_camp)
        if my is None or enemy is None:
            return None
        last_hit, last_hit_status = sdk_last_hit(state)
        return {"my": my, "en": enemy,
                "frame_no": getattr(req_pb, "frame_no", None),
                "sdk_last_hit": last_hit, "sdk_last_hit_status": last_hit_status,
                "my_organs": organ_snapshot(req_pb, my_camp),
                "en_organs": organ_snapshot(req_pb, en_camp)}

    def collect_trajectory(self, max_decisions=None, deterministic=False,
                           random_policy=False, matchup=None, evaluation=False):
        from .rollout import collect_episode
        return collect_episode(self, max_decisions, deterministic, random_policy,
                               matchup, evaluation)

    def _outcome_evidence(self, state, outcome):
        req = state.get("req_pb")
        return {"outcome": outcome, "my_camp": int(self._camps[0]) if self._camps[0] else None,
                "frame_no": int(getattr(req, "frame_no", -1)),
                "snapshot": self._take_snapshot(state), "control": self.control,
                **self.episode_info}

    def build_observation(self, state):
        """在 705 维观测末尾追加我方英雄身份 one-hot，得到 725 维输入。"""
        observation = np.asarray(state["observation"], dtype=np.float32).reshape(-1)
        if observation.shape == (725,):
            return observation
        if observation.shape != (705,):
            raise ValueError(f"SDK 原始观测应为 705 维，实际 {observation.shape}")
        req = state.get("req_pb")
        player_id = state.get("player_id")
        hero_id = None
        if req is not None:
            for hero in req.hero_list:
                if hero.runtime_id == player_id:
                    hero_id = int(hero.config_id)
                    break
        if hero_id not in HERO_ID_INDEX_DICT:
            raise ValueError(f"无法编码我方英雄身份 {hero_id}，请核对英雄池和 SDK")
        identity = np.zeros(HERO_IDENTITY_DIM, dtype=np.float32)
        identity[HERO_ID_INDEX_DICT[hero_id]] = 1.0
        return np.concatenate([observation, identity])

    def reset(self, matchup=None, evaluation=False) -> Dict:
        """结束旧局、重建 ZMQ、安排双方阵容，并从首帧初始化奖励快照。"""
        close_started = time.monotonic()
        self._close_previous_game()
        close_seconds = time.monotonic() - close_started
        start_started = time.monotonic()
        for addr in self.env.addrs:
            try:
                self.env.lib_processor.server_manager.Delete(addr)
            except Exception:
                logger.debug('删除 zmq server %s 失败（忽略）', addr)
        if matchup is not None:
            self.player_id = matchup.player_id
            self.current_matchup = matchup
            camp_config = matchup.camp_config()
        elif self.camp_config is not None:
            camp_config = self.camp_config
            self.current_matchup = None
        else:
            self.current_matchup = self.schedule.next()
            self.player_id = self.current_matchup.player_id
            camp_config = self.current_matchup.camp_config()
        self.use_common_ai = [i != self.player_id for i in range(2)]
        self.control = None
        self.episode_info = {"started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                             "runtime_id": self.env.runtime_id, "simulator_mode": "remote"}
        logger.info("开局 start=%s runtime=%s player_index=%s heroes=%s common_ai=%s",
                    self.episode_info["started_at"], self.env.runtime_id, self.player_id,
                    camp_config["heroes"], self.use_common_ai)
        self._game_running = True
        _obs, _reward, _done, states = self.env.reset(
            camp_config, use_common_ai=self.use_common_ai, eval=evaluation,
        )
        self._step_count = 0
        self._camps = (None, None)
        self._validate_control(states[self.player_id], camp_config)
        self.episode_info.update(ready_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                 **state_game_ids(states[self.player_id]),
                                 initial_frame=int(getattr(states[self.player_id]["req_pb"], "frame_no", -1)))
        logger.info("控制方已核验 sdk_client_id=%s sdk_game_id=%s hero=%s camp=%s runtime_player_id=%s",
                    self.episode_info["sdk_client_id"], self.episode_info["sdk_game_id"],
                    self.control["hero_id"], self.control["camp"], self.control["runtime_player_id"])
        self._prev_snapshot = self._take_snapshot(states[self.player_id])
        self._last_snapshot = self._prev_snapshot
        self._game_running = True
        self._game_done = False
        self.reset_timings = {"close_game_seconds": close_seconds,
                              "start_game_seconds": time.monotonic() - start_started}
        return states[self.player_id]

    def step(self, action6):
        """发送我方动作，对手由 common AI 控制；返回状态差分奖励。"""
        actions = [NOOP_ACTION] * self.cfg.env.player_num
        actions[self.player_id] = tuple((int(x) for x in action6))
        (_obs, reward, done, states) = self.env.step(actions)
        self._step_count += 1
        self._validate_control(states[self.player_id])
        snap = self._take_snapshot(states[self.player_id])
        self._last_snapshot = snap
        if snap is None:
            components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            reward_scalar = 0.0
        elif self._prev_snapshot is None:
            self._prev_snapshot = snap
            components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            reward_scalar = 0.0
        else:
            components = state_reward_terms(self._prev_snapshot, snap, self.cfg.reward)
            reward_scalar = float(sum(components.values()))
            self._prev_snapshot = snap
        time_penalty = -self.cfg.env.reward_time_coef
        reward_scalar += time_penalty
        components['time'] = time_penalty
        done_bool = bool(done[self.player_id])
        if done_bool:
            self._game_running = False
            self._game_done = True
        return (states[self.player_id], float(reward_scalar), done_bool, components)

    def _validate_control(self, state, camp_config=None):
        """把接收动作的 SDK 槽位、观测 player_id、英雄和阵营对齐后才计算奖励。"""
        if not state or state.get("req_pb") is None:
            raise RuntimeError("控制方校验失败：训练槽位没有观测")
        runtime_player_id = state.get("player_id")
        if runtime_player_id is None or runtime_player_id != self.env.player_list[self.player_id]:
            raise RuntimeError("控制方校验失败：观测 player_id 与 SDK player_list 不一致")
        heroes = list(state["req_pb"].hero_list)
        mine = [h for h in heroes if h.runtime_id == runtime_player_id]
        if len(mine) != 1:
            raise RuntimeError("控制方校验失败：不能唯一确定受控英雄")
        mine = mine[0]
        camp = int(mine.camp)
        enemies = [h for h in heroes if int(h.camp) != camp]
        if len(enemies) != 1 or camp != self.player_id + 1:
            raise RuntimeError("控制方校验失败：实际阵营与训练槽位不一致")
        enemy = enemies[0]
        if int(enemy.camp) != 2-self.player_id:
            raise RuntimeError("控制方校验失败：对手阵营无效")
        actual = {"player_index": self.player_id, "runtime_player_id": int(runtime_player_id),
                  "hero_id": int(mine.config_id), "opponent_id": int(enemy.config_id),
                  "camp": camp, "opponent_camp": int(enemy.camp),
                  "common_ai": self.use_common_ai.copy(), "verified": True}
        if camp_config is not None:
            expected = [side[0]["hero_id"] for side in camp_config["heroes"]]
            if actual["hero_id"] != expected[self.player_id] or actual["opponent_id"] != expected[1-self.player_id]:
                raise RuntimeError(f"控制方校验失败：实际英雄 {actual} 与请求阵容 {expected} 不一致")
        if hasattr(self.env, "need_predict") and any(
                bool(self.env.need_predict(i)) != (i == self.player_id) for i in range(2)):
            raise RuntimeError("控制方校验失败：SDK 动作槽位被交给 common AI")
        if self.control is not None and actual != self.control:
            raise RuntimeError("控制方校验失败：对局中受控英雄或阵营发生变化")
        for key, current in state_game_ids(state).items():
            previous = self.episode_info.get(key)
            if previous and current != previous:
                raise RuntimeError(f"控制方校验失败：对局中 {key} 发生变化")
        self.control = actual
        self._camps = camp, int(enemy.camp)

    def _resolve_camps(self, req_pb):
        my_rid = self.env.player_list[self.player_id]
        if my_rid is None:
            return (None, None)
        my_camp = None
        for hero in req_pb.hero_list:
            if getattr(hero, 'runtime_id', None) == my_rid:
                my_camp = hero.camp
                break
        if my_camp is None:
            return (None, None)
        for hero in req_pb.hero_list:
            if hero.camp != my_camp:
                return (my_camp, hero.camp)
        return (None, None)

    def _close_previous_game(self):
        """先等自然收尾，超时再 stop；确认旧进程退出后才允许重建连接。"""
        if not (self._game_done or self._game_running):
            return
        if self._game_done:
            # 终局观测早于模拟器退出；立即启动同 runtime_id 会杀掉旧进程，截断 ABS/stat。
            if self._wait_for_game_exit(NATURAL_EXIT_TIMEOUT):
                self._game_done = self._game_running = False
                return
            logger.warning("终局后 %.0f 秒 gamecore 仍未退出，尝试 stop 并确认退出；"
                           "本局回放可能不完整，客户端=%s",
                           NATURAL_EXIT_TIMEOUT, self.episode_info.get("sdk_client_id", ""))
        # 超时后的 finally 清理也直接重试 stop，不再重复等待自然终局。
        self._game_done, self._game_running = False, True
        self._stop_game_quietly()
        if not self._wait_for_game_exit(STOP_EXIT_TIMEOUT):
            raise RuntimeError("stop_game 后 gamecore 仍未退出；为避免旧局占用连接，停止训练并保存模型")
        self._game_running = False

    def _wait_for_game_exit(self, timeout):
        """只轮询 HTTP 状态，不调用可能无限等待下一帧的 native 接收。"""
        deadline = time.monotonic() + timeout
        while self.env.game_launcher.check_exists_game(self.env.runtime_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
        return True

    def _stop_game_quietly(self):
        try:
            self.env.game_launcher.stop_game(self.env.runtime_id)
        except Exception as exc:
            logger.warning('stop_game 失败（可忽略）：%s', exc)

    def close(self):
        self._close_previous_game()

    @torch.no_grad()
    def sample_action(self, feature, legal_action_172, lstm_state,
                      deterministic=False, random_policy=False):
        """先采样 button，再用该 button 对应的 mask 采样 target。

        采样和 learner 共用 masked_softmax / probs_to_log_probs，保证概率一致。
        random_policy 仅供诊断，使用时也记录真正的均匀采样概率。
        """
        feature = torch.as_tensor(feature, dtype=torch.float32, device=self.device)
        output = self.inference(feature.reshape(1, 1, -1), lstm_state)
        legal = torch.as_tensor(legal_action_172, device=self.device)
        masks = split_legal_action_expanded(legal)
        actions, log_probs = [], []
        capture_probabilities = getattr(self, "capture_action_probabilities", False)
        head_probabilities = [] if capture_probabilities else None
        for head in range(6):
            mask = masks[head] if head < 5 else masks[5][actions[0]]
            logits = output["logits_list"][head].reshape(1, -1)
            if random_policy:
                logits = torch.zeros_like(logits)
            probs = masked_softmax(logits, mask.reshape(1, -1))
            if head == 0:
                self.last_button_probabilities = probs.detach().cpu().numpy().reshape(-1)
            if capture_probabilities:
                head_probabilities.append(probs.detach().cpu().reshape(-1).tolist())
            choice = sample_from_probs(probs, deterministic=deterministic)
            probability = probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            actions.append(int(choice.item()))
            log_probs.append(float(probs_to_log_probs(probability).item()))
        if capture_probabilities:
            self.last_head_probabilities = head_probabilities
        return (np.asarray(actions, dtype=np.int64),
                np.asarray(log_probs, dtype=np.float32),
                output["lstm_state_new"], float(output["value"].item()))

    def _terminal_outcome(self, state: Dict) -> Optional[str]:
        try:
            req_pb = state.get('req_pb')
            if req_pb is None:
                return None
            (my_camp, en_camp) = self._camps
            if my_camp is None:
                (my_camp, en_camp) = self._resolve_camps(req_pb)
                if my_camp is None:
                    return None
            outcome = determine_outcome(req_pb, my_camp, en_camp)
            if outcome is not None:
                return outcome
        except Exception as exc:
            logger.debug('terminal_outcome 判定失败，忽略终端奖励：%s', exc)
        return None
