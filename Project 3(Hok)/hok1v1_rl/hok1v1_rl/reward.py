"""HoK 1v1 奖励解析、dense reward 与 GAE 计算。

────────────────────────────────────────────────────────────────────────────
奖励向量的来源与长度（实测确认，与官方假设不同）
────────────────────────────────────────────────────────────────────────────
本项目的 gamecore（hok_env_gamecore_20260420）实测：

    原始 ``states[i]["reward"]``              = **10 维**
    ``_state2ret`` 返回的 ``reward[i]``       = **9 维**
    关系已实测验证： return == raw[:6] + raw[7:]   （丢弃 raw[6]）

而官方 hok_env 的假设是 raw 9 维 → 返回 8 维。也就是说**这个 gamecore 的
reward 向量比官方多了一个分量**，因此不能沿用"返回 8 维"的硬假设。

8 个权重（= 官方 config.json）在官方语义下对应返回向量的 index 0..7：

    [0] money           金钱变化
    [1] exp             经验变化
    [2] hp_point        英雄血量变化
    [3] ep_rate         能量/技能值变化
    [4] kill            击杀数
    [5] dead            死亡数
    [6] tower_hp_point  防御塔血量变化
    [7] last_hit        补刀数
    [8] 复合分量        已查明（见下）：gamecore 输出的加权复合奖励，
                        是其他分量的线性组合，不含独立信息，不计入 dense reward

⚠️ 分量语义与预乘结论（logs/metrics.jsonl 158 条训练日志交叉验证 + 探针对局）：

1. 分量确实已被 config.json 权重预乘。实测 dead 每死亡 -1.0（= -1.0 × 1 次）、
   kill 每击杀 -0.5（118 条 kill 日志全部是 0.5 的倍数；仓库 config.json 写的
   -0.6 与 Windows 侧 gamecore 实际不符，以实测 -0.5 为准，见 OFFICIAL_WEIGHTS）。
2. 各分量的统计口径（同口径分量在日志中的符号表现一致）：
   - 绝对量（我方自身）：money / ep 变化量，hp / tower 血量变化量，
     kill / dead 事件次数
   - 差值量（1v1 零和优势）：exp、last_hit 恒为负 => 我方与敌方之差
3. 训练奖励已改用 **req_pb 状态差分**（``compute_state_reward``），语义精确：
   英雄 hp/money/exp/killCnt/deadCnt 与塔/水晶 hp 都从 req_pb 原始字段读取。
   ``compute_dense_reward``（reward 向量去加权路径）仅保留给诊断脚本。
   原因（probe_reward.py 实测）：reward 向量的 dead 分量把小兵死亡也算进去
   （双方英雄 deadCnt=0 的一局触发了 11 次 dead 事件），kill 权重与仓库
   config.json 不符（Windows 侧被改过），且胜负无法从向量判定。

下面的 ``parse_env_reward`` 只解析前 8 个已知分量，第 9 个分量（复合奖励，
不含独立信息）保留但不计入 dense reward。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

#: 已确认语义的分量个数（对应官方 config.json 的 8 个权重）。
KNOWN_REWARD_DIM = 8

#: 8 个已知分量的名字（顺序与 parse_env_reward 的返回一致）。
REWARD_COMPONENT_NAMES = (
    "money", "exp", "hp", "ep", "kill", "dead", "tower", "last_hit",
)


#: gamecore 侧 config.json 的官方权重（分量已被这些权重预乘，实测确认）。
#: 去加权（deweight_components）时逐分量除以这些值，还原原始单位。
#: 注意 kill 用实测的 -0.5（仓库 config.json 副本写的是 -0.6，与 Windows 侧
#: gamecore 实际不一致；logs/metrics.jsonl 的 kill 分量全是 0.5 的倍数）。
OFFICIAL_WEIGHTS = {
    "money": 0.006,
    "exp": 0.006,
    "hp": 2.0,
    "ep": 0.75,
    "kill": -0.5,
    "dead": -1.0,
    "tower": 5.0,
    "last_hit": 0.5,
}


def deweight_components(c: Dict[str, float]) -> Dict[str, float]:
    """把已按官方权重预乘的分量还原为原始单位。

    各分量统计口径（logs/metrics.jsonl 交叉验证，见模块 docstring）：
    - 绝对量（我方自身）：money / ep 变化量、hp / tower 血量变化量、
      kill / dead 事件次数
    - 差值量（1v1 零和优势）：exp、last_hit（我方与敌方之差）
    """
    return {k: float(c[k]) / OFFICIAL_WEIGHTS[k] for k in REWARD_COMPONENT_NAMES}


#: ────────────────────────────────────────────────────────────────────────
#: 归一化基准：所有 dense 分量都换算成**可解释的单位**再加权。
#: 这样权重的绝对量级不再依赖 gamecore 的原始数值范围，调参也变得可读。
#: ────────────────────────────────────────────────────────────────────────
#: 塔 5000 + 水晶 7000（req_pb organ_list 实测 max_hp）
ORGAN_UNIT = 12000.0
#: 金钱 / 经验：1000 记为 1 个单位
MONEY_UNIT = 1000.0
EXP_UNIT = 1000.0
#: max_hp 取不到时的兜底（英雄 1 级满血约 3000）
HP_UNIT_FALLBACK = 3000.0


@dataclass
class RewardWeights:
    """dense 奖励权重（作用在**归一化单位**上，见下面的量纲表）。

    ────────────────────────────────────────────────────────────────────────
    ⚠️ 为什么推翻了"照搬官方 config.json 权重"的做法
    ────────────────────────────────────────────────────────────────────────
    上一版曾把权重直接改成官方值（tower 5.0 / hp 2.0 / money 0.006）。
    跑真实数据后发现**行不通**：实测（`scripts/probe_components.py`，
    完整一局、随机策略）

        hp 旧口径（Δhp 差分）      = +330,011   ← 65.5 "个血条"
        hp 可见性安全口径          =  +32,902
        hp 伤害计数器口径          =   -3,511   ← 真实值，0.70 "个血条"
        塔/水晶                    =  -12,000   ← 1.00 "座基地"

    也就是说旧 hp 项的 **94% 是视野占位值造成的幽灵伤害**（见
    `extract_hero_snapshot` 的说明）。官方 2.0 的权重是针对 gamecore
    内部那个 hp_point 量标定的，直接乘到 req_pb 的原始 hp 上，
    只会把幽灵伤害放大 2000 倍。

    因此这里改为**先归一化再乘权重**：
      * hp    -> (对英雄造成的伤害 - 受到英雄的伤害) / max_hp  [个血条]
      * tower -> ((我方塔+水晶掉血) - (敌方塔+水晶掉血)) / 12000  [座基地]
      * money -> 我方金钱变化 / 1000
      * exp   -> 我方经验变化 / 1000
      * kill / dead -> 次数（本身就是无量纲）

    ────────────────────────────────────────────────────────────────────────
    权重取值理由（目标：**结局 > 基地 > 换血**）
    ────────────────────────────────────────────────────────────────────────
    一局最多推进 1 座基地（tower=1.0）→ 10.0；
    换血一个血条（hp=1.0）→ 1.0；被杀一次 -2.0（约等于两个血条，合理）；
    赢/输/截断 ±50 —— 比"推掉整座基地"（10）还大 5 倍，
    保证**胜负面永远压过任何 dense 项**。

    实测一局的 dense 合计（输局）约 `-13`，加终端 -50 => 约 -63；
    赢局约 +62。落差 ~125，且符号与真实胜负一致 —— 这是旧奖励设计
    最缺的性质（旧设计输局也能拿到 +72 的正回报）。
    """
    money: float = 0.5
    exp: float = 0.5
    hp: float = 1.0
    ep: float = 0.0          # 已停用：ep 惩罚"放技能"，与取胜目标冲突
    kill: float = 2.0
    dead: float = -2.0
    tower: float = 10.0
    last_hit: float = 0.0    # 仅旧 reward 向量路径（诊断）使用
    win: float = 50.0
    lose: float = -50.0
    truncate: float = -50.0


DEFAULT_WEIGHTS = RewardWeights()


def parse_env_reward(reward_vec) -> Dict[str, float]:
    """把 env 返回的 reward 解析为具名 dict。

    Args:
        reward_vec: 长度 >= 8 的 array-like（本项目实测为 9）
    Returns:
        dict: money, exp, hp, ep, kill, dead, tower, last_hit
              （若长度 >= 9，额外带一个 ``extra`` 键：gamecore 的加权复合
              奖励，是其他分量的线性组合，不计入 dense reward）
    """
    r = np.asarray(reward_vec, dtype=np.float32).reshape(-1)
    assert r.shape[0] >= KNOWN_REWARD_DIM, (
        f"reward 向量过短：{r.shape}（期望 >= {KNOWN_REWARD_DIM}）"
    )
    out = {
        "money": float(r[0]),
        "exp": float(r[1]),
        "hp": float(r[2]),
        "ep": float(r[3]),
        "kill": float(r[4]),
        "dead": float(r[5]),
        "tower": float(r[6]),
        "last_hit": float(r[7]),
    }
    if r.shape[0] > KNOWN_REWARD_DIM:
        # 第 9 个分量已查明：gamecore 输出的加权复合奖励。实测（探针对局
        # 300 帧最小二乘拟合，R²=0.9962）:
        #   raw[9] = 0.75*exp + 2.0*ep + 0.006*(hp+tower) + 0.5*dead + 微小残差
        # 即它是其他分量的线性组合（且权重与 config.json 语义错位，例如
        # hp 的 2.0 权重乘到了 ep 分量上），不含独立信息。继续默认不计入
        # dense reward，保留在 dict 里仅作观测。
        out["extra"] = float(r[KNOWN_REWARD_DIM])
    return out


def compute_dense_reward(reward_vec, terminal: Optional[str] = None,
                         weights: RewardWeights = DEFAULT_WEIGHTS
                         ) -> Tuple[float, Dict[str, float]]:
    """计算单步标量 dense reward（新奖励设计）。

    流程：解析 9 维分量 -> 按 OFFICIAL_WEIGHTS 去加权还原原始单位 ->
    乘 RewardWeights 求和。

    Args:
        reward_vec: 9 维环境 reward（_state2ret 切分后的，gamecore 已按
            官方权重预乘）
        terminal:   None 表示对局中；"win" / "lose" 表示终端步
        weights:    RewardWeights
    Returns:
        (scalar_reward, component_dict)
    """
    c = parse_env_reward(reward_vec)
    raw = deweight_components(c)
    r = (
        weights.money * raw["money"]
        + weights.exp * raw["exp"]
        + weights.hp * raw["hp"]
        + weights.ep * raw["ep"]
        + weights.kill * raw["kill"]
        + weights.dead * raw["dead"]
        + weights.tower * raw["tower"]
        + weights.last_hit * raw["last_hit"]
    )
    # 终端奖励只在 episode 结束时叠加一次，避免每步重复奖励
    if terminal == "win":
        r += weights.win
    elif terminal == "lose":
        r += weights.lose
    return float(r), c


def _enum_value(v) -> int:
    """取枚举的数值（IntEnum 与普通 enum 通用）。"""
    return v.value if hasattr(v, "value") else v


def _num_of(obj, name: str, default=None):
    """安全读取 protobuf 字段并转 float。

    字段缺失或不可转数值时返回 ``default``（而不是抛异常）：
    gamecore 不同版本对可选字段的填充不一致，缺字段不应该让整局训练崩掉。
    """
    v = getattr(obj, name, None)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _camp_of(x, default=-1):
    """取对象的 camp 数值（IntEnum / 普通 enum 通用）。"""
    c = getattr(x, "camp", None)
    if c is None:
        return default
    try:
        return int(c)
    except (TypeError, ValueError):
        return int(getattr(c, "value", default))


def _visible_to(req_pb, observer_camp, target_camp) -> Optional[bool]:
    """``target_camp`` 阵营对 ``observer_camp`` 是否可见。

    ⚠️ 这个函数是本次奖励修正的核心。实测（probe_diag.py）：
    ``hero.camp_visible`` 是**按观察者**给出的 2 元列表，
    对 camp C 的英雄，``camp_visible[j]`` 表示 "camp(j+1) 对 C 是否可见"。

    而**英雄不可见时，它的 `hp` 字段是一个占位值（实测为 1）**：
    实测 f40/f80 敌方 hp=1（max_hp=3150），f120 才变成 3150。
    旧实现无脑读 `hero.hp` 做差分，于是"敌人脱离视野"（3150 -> 1）
    被记成"敌人挨了 3149 点伤害"，白送奖励。

    实测一局累计：旧口径 +330,011（65.5 个血条），真实伤害计数器只有
    -3,511（0.70 个血条）—— **94% 是幽灵伤害**，而且奖励"让敌人从视野里消失"
    等于奖励逃跑，与取胜方向完全相反。
    """
    if observer_camp is None or target_camp is None:
        return None
    for h in req_pb.hero_list:
        if _camp_of(h) == observer_camp:
            cv = getattr(h, "camp_visible", None)
            if cv is None:
                return None
            try:
                i = int(target_camp) - 1
                if 0 <= i < len(cv):
                    return bool(cv[i])
            except (TypeError, ValueError):
                return None
            return None
    return None


def extract_hero_snapshot(req_pb, camp,
                          observer_camp: Optional[int] = None
                          ) -> Optional[Dict[str, float]]:
    """从 req_pb 提取指定阵营英雄的状态快照。

    字段实测（probe_diag.py dump）：
      * ``hp`` / ``max_hp``：当前/最大血量（max_hp 随升级增长：
        实测 3150 -> 3425 -> 3700 -> ... -> 8708）
      * ``totalHurtToHero``：累计"我对英雄造成的伤害"（单调递增）
      * ``totalBeHurtByHero``：累计"我受到英雄的伤害"（单调递增）
      * ``money`` / ``exp`` / ``ep`` / ``killCnt`` / ``deadCnt``
      * ``camp_visible``：按观察者的可见性（见 ``_visible_to``）

    ⚠️ **不要直接用 hp 做差分**：英雄不可见时 hp 是占位值（1）。
    伤害计数器（``hurt_to_hero`` / ``hurt_by_hero``）与视野无关且单调，
    是换血项的正确来源。

    Args:
        req_pb:        环境 state 里的 req_pb
        camp:          要提取的阵营
        observer_camp: 观察者阵营（用于判断可见性）；None 表示不判可见性
    Returns:
        dict，或英雄缺席时 None
    """
    for hero in req_pb.hero_list:
        if _camp_of(hero) != camp:
            continue
        return {
            "hp": _num_of(hero, "hp", 0.0),
            "max_hp": _num_of(hero, "max_hp", None),
            "money": _num_of(hero, "money", 0.0),
            "exp": _num_of(hero, "exp", 0.0),
            "ep": _num_of(hero, "ep", 0.0),
            "kills": _num_of(hero, "killCnt", 0.0),
            "deaths": _num_of(hero, "deadCnt", 0.0),
            # 与视野无关的累计伤害计数器（换血项的正确来源）
            "hurt_to_hero": _num_of(hero, "totalHurtToHero", None),
            "hurt_by_hero": _num_of(hero, "totalBeHurtByHero", None),
            "visible": _visible_to(req_pb, observer_camp, camp),
        }
    return None


#: organ.type 实测取值（probe_reward.py dump）
ORGAN_TYPE_TOWER = 21      # ACTOR_TOWER，max_hp 5000
ORGAN_TYPE_CRYSTAL = 24    # ACTOR_CRYSTAL，max_hp 7000
#: 23 = ACTOR_TOWER_SPRING（泉水塔），不计入推塔/胜负判定


def find_organ_hp(req_pb, camp, organ_type: int) -> Optional[float]:
    """查找指定阵营、指定类型建筑的血量。

    ⚠️ 与 ``extract_organ_hp`` 的关键区别：**缺席返回 None，而不是 0.0**。

    这个区别决定了胜负判定是否可信：建筑被推掉时 hp 就是 0，
    而该建筑那一帧不在 ``organ_list`` 里时 hp 也是"没有"。
    若统一折成 0.0，一次缺帧就会被 ``determine_outcome`` 读成
    "水晶被打爆" —— 既可能把赢局判成输局（误差 ±10 终端奖励），
    也会让训练奖励出现凭空的推塔净额。
    """
    for o in req_pb.organ_list:
        if o.camp != camp:
            continue
        if _enum_value(o.type) == organ_type:
            return float(o.hp)
    return None


def extract_organ_hp(req_pb, camp) -> Tuple[float, float]:
    """该阵营塔 + 水晶的血量（推塔奖励的差分基准）。

    organ.type 实测：21=ACTOR_TOWER（塔，max_hp 5000）、24=ACTOR_CRYSTAL
    （水晶，max_hp 7000）、23=ACTOR_TOWER_SPRING（泉水塔，不计入）。
    对局结束时被推方的塔与水晶 hp 均为 0（probe_reward.py 实测）。

    这里把"缺席"折成 0.0，只是为了给差分奖励一个稳定的数值基准
    （缺席帧造成的瞬时跳变由调用方通过 snap is None 兜底过滤）。
    **胜负判定不要用这个函数**，请用 find_organ_hp，见其 docstring。
    """
    tower_hp = find_organ_hp(req_pb, camp, ORGAN_TYPE_TOWER)
    crystal_hp = find_organ_hp(req_pb, camp, ORGAN_TYPE_CRYSTAL)
    return (
        0.0 if tower_hp is None else tower_hp,
        0.0 if crystal_hp is None else crystal_hp,
    )


def hp_term(prev: Dict, cur: Dict) -> float:
    """换血项（归一化到"个血条"）。

    首选**累计伤害计数器**：``totalHurtToHero`` / ``totalBeHurtByHero``
    单调递增且与视野无关，是唯一可靠的换血来源。

    退路：若 goalcore 没给这两个字段，则退回 hp 差分，但**要求双方
    在前后两帧都可见**（``visible``），否则返回 0 —— 这样至少不会把
    "敌人脱离视野"记成伤害。注意退路仍然会被"升级导致 max_hp 变化"
    之类的跳变污染，所以只要计数器可用就一定走计数器。
    """
    my_p, my_c = prev["my"], cur["my"]
    en_p, en_c = prev["en"], cur["en"]

    unit = my_c.get("max_hp") or HP_UNIT_FALLBACK
    if not unit or unit <= 0:
        unit = HP_UNIT_FALLBACK

    ctr_keys = ("hurt_to_hero", "hurt_by_hero")
    if all(my_p.get(k) is not None for k in ctr_keys) and \
       all(my_c.get(k) is not None for k in ctr_keys):
        dealt = max(0.0, my_c["hurt_to_hero"] - my_p["hurt_to_hero"])
        taken = max(0.0, my_c["hurt_by_hero"] - my_p["hurt_by_hero"])
        return (dealt - taken) / unit

    # 退路：hp 差分 + 可见性门槛
    if not (my_p.get("visible") and my_c.get("visible")
            and en_p.get("visible") and en_c.get("visible")):
        return 0.0
    d_en = en_c["hp"] - en_p["hp"]
    d_my = my_c["hp"] - my_p["hp"]
    return (max(0.0, -d_en) - max(0.0, -d_my)) / unit


def compute_state_reward(prev: Dict, cur: Dict,
                         weights: RewardWeights = DEFAULT_WEIGHTS) -> float:
    """由相邻两步的英雄/塔状态快照计算 dense reward（训练路径）。

    prev/cur 结构（env_runner._take_snapshot 产出）::

        {"my": extract_hero_snapshot(...), "en": ...,
         "my_organ": 我方塔+水晶总血量, "en_organ": 敌方塔+水晶总血量}

    返回值等于 ``sum(state_reward_terms(...).values())``，
    即与 ``env_runner`` 逐分量记录的 dict **严格自洽**。

    各项都先归一化成可解释单位（见 ``RewardWeights`` 的说明）：
    - hp：换血净额 / max_hp   [个血条]，走累计伤害计数器（不依赖视野）
    - tower：推塔净额 / 12000 [座基地]
    - money / exp：我方变化 / 1000
      注意用**我方绝对值**而不是与敌方的差值：敌方英雄不可见时
      它的 money/exp 也可能读到占位值，做差会引入同类幽灵信号。
      被动收入带来的漂移在这个量级下（~1.5）远小于基地(10)与胜负(50)，
      不足以诱发"拖局"。
    - kill / dead：次数
    """
    return float(sum(state_reward_terms(prev, cur, weights).values()))


def state_reward_terms(prev: Dict, cur: Dict,
                       weights: RewardWeights = DEFAULT_WEIGHTS) -> Dict[str, float]:
    """把 dense reward 拆成**加权后的逐分量贡献**。

    与 ``compute_state_reward`` 共用同一套公式，因此逐分量之和恒等于
    标量奖励（``env_runner.step`` 依赖这个不变式）。

    值已乘权重且已归一化，所以各分量可以直接横向比较"谁在主导奖励"：
    例如 ``tower = -10.0`` 就是字面意思"丢了一整座基地"。
    """
    my_p, my_c = prev["my"], cur["my"]

    d_money = (my_c["money"] - my_p["money"]) / MONEY_UNIT
    d_exp = (my_c["exp"] - my_p["exp"]) / EXP_UNIT
    d_ep = (my_c["ep"] - my_p["ep"]) / max(1.0, my_c.get("max_hp") or HP_UNIT_FALLBACK)

    # 塔/水晶：正值 = 我方推得更狠
    d_organ = ((cur["my_organ"] - prev["my_organ"])
               - (cur["en_organ"] - prev["en_organ"])) / ORGAN_UNIT

    return {
        "money": weights.money * d_money,
        "exp": weights.exp * d_exp,
        "hp": weights.hp * hp_term(prev, cur),
        "ep": weights.ep * d_ep,
        "kill": weights.kill * (my_c["kills"] - my_p["kills"]),
        "dead": weights.dead * (my_c["deaths"] - my_p["deaths"]),
        "tower": weights.tower * d_organ,
        "last_hit": 0.0,
    }


def determine_outcome(req_pb, my_camp, en_camp) -> Optional[str]:
    """对局结束时按塔/水晶血量判定胜负。

    实测（probe_reward.py）：对局结束条件是被推方的塔+水晶双双摧毁，
    而官方 actor.py 用"我方英雄 hp > 0"判胜，在双方英雄都活着的推塔局里
    恒定误判为 win（历史日志 win_rate 恒为 1.0 的来源，±10 终端奖励
    因此一直在奖励输局）。

    判定顺序：我方水晶归零 => lose；敌方水晶归零 => win；
    其余结束方式（时间到等）比较双方塔+水晶总血量，比不出返回 None。

    ⚠️ 必须用 ``find_organ_hp``（缺席 -> None）而不是 ``extract_organ_hp``
    （缺席 -> 0.0）。建筑被推掉时 hp 为 0，建筑不在 organ_list 里时
    同样是"读不到"；若把后者也当成 0，任何一次缺帧都会让本函数
    判定"水晶被打爆"，进而把 ±10 终端奖励发给错误的胜负。
    """
    my_crystal = find_organ_hp(req_pb, my_camp, ORGAN_TYPE_CRYSTAL)
    en_crystal = find_organ_hp(req_pb, en_camp, ORGAN_TYPE_CRYSTAL)

    # 只有"确实读到了水晶且血量归零"才算被推
    if my_crystal is not None and my_crystal <= 0.0:
        return "lose"
    if en_crystal is not None and en_crystal <= 0.0:
        return "win"

    # 兜底：比较双方塔+水晶总血量。
    # ⚠️ 只能**同类相比较**：若某一方缺塔、另一方有塔，把两边直接求和
    #    会让"少一个建筑"变成血量劣势，凭空翻转胜负。
    #    因此只保留两边都读到的建筑类型；一类都没有则返回 None（不发终端奖励）。
    my_tower = find_organ_hp(req_pb, my_camp, ORGAN_TYPE_TOWER)
    en_tower = find_organ_hp(req_pb, en_camp, ORGAN_TYPE_TOWER)
    pairs = [
        (a, b) for a, b in ((my_tower, en_tower), (my_crystal, en_crystal))
        if a is not None and b is not None
    ]
    if not pairs:
        return None

    my_total = float(sum(a for a, _ in pairs))
    en_total = float(sum(b for _, b in pairs))
    if my_total > en_total:
        return "win"
    if en_total > my_total:
        return "lose"
    return None


def compute_value_targets(rewards, gamma: float, values, last_value: float,
                          lamda: float, dones=None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """计算 GAE 优势与 returns。

    GAE (Generalized Advantage Estimation)::

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lamda * (1 - done_t) * A_{t+1}
        returns = A_t + V(s_t)

    反向递推实现，复杂度 O(T)。

    ⚠️ **done 掩码是必需的**（原实现把 ``not_done`` 硬编码为 1.0 并留了 TODO）：
    若不切断，终局之后的 bootstrap value 会跨 episode 泄漏回上一局，
    使优势估计系统性偏移。传入的 ``dones[t] == 1`` 表示"第 t 步之后该轨迹已终止
    （或被截断）"，此时不再使用 V(s_{t+1})，也不继续累加 GAE。

    关于"截断"（time limit）与"终止"（terminal）的区别：
    严格来说截断时应当继续 bootstrap（因为状态不是真正的终止态）。
    本实现与官方 baseline 一致，统一按"切断"处理；
    若要区分，可另加 ``truncated`` 数组并在截断处保留 bootstrap。

    Args:
        rewards:    (T,) 每步标量奖励
        gamma:      折扣因子
        values:     (T,) 每步 V(s_t)
        last_value: 最后一步 bootstrap 用的 V(s_T)
        lamda:      GAE lambda
        dones:      (T,) 或 None。None 视为全程未终止（仅适用于单段连续轨迹）
    Returns:
        (advantages, returns)，均与 rewards 等长
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    T = len(rewards)
    assert T == len(values), f"rewards/values 长度不一致：{T} vs {len(values)}"

    if dones is None:
        # 显式提示：默认"未终止"只适合已按 episode 切好的单段轨迹。
        dones_arr = np.zeros(T, dtype=np.float32)
    else:
        dones_arr = np.asarray(dones, dtype=np.float32).reshape(-1)
        assert len(dones_arr) == T, f"dones 长度 {len(dones_arr)} != T {T}"

    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        # done_t == 1 时切断：不再 bootstrap，也不再累积后续 GAE
        not_done = 1.0 - dones_arr[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lamda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class RunningRewardNormalizer:
    """回报的运行均值-方差归一化器（Welford 在线算法）。

    用途：新奖励设计下单局回报绝对值可达数百（kill ±2/次、终端 ±10、
    tower 单局 ±4~14 累积），直接作为 value 回归目标会让 V 网络追一个
    跨 batch 漂移的尺度，vloss 不稳。
    典型用法：每个 batch 先 ``update(returns)``，再用 ``normalize(returns)``
    作为 value 网络的回归目标，使目标始终是零均值单位方差分布。
    由 ``learner.PPOLearner.update`` 在 ``normalize_returns`` 开启时调用。

    使用 Welford 在线算法，避免存储全部历史数据。
    """

    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps      # 初始化为 eps 而非 0，避免除零
        self.eps = eps

    def update(self, rewards) -> None:
        """用一批新数据在线更新均值/方差（Welford 并行合并公式）。"""
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if r.size == 0:
            return
        batch_mean = r.mean()
        batch_var = r.var()
        batch_count = r.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (
            m_a + m_b + delta * delta * self.count * batch_count / tot_count
        ) / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """用当前统计量归一化（不更新统计量）。"""
        return (np.asarray(x) - self.mean) / (np.sqrt(self.var) + self.eps)
