"""从相邻状态计算奖励；缺失数据表示未知，不能当成死亡或掉血。

moneyCnt 是官方 actor 使用的累计收入，money 是钱包；经验默认不启用。
所有日志分量均为加权后的奖励贡献，与总奖励直接相加对应。
"""
from typing import Dict, Optional, Tuple
import numpy as np
from .config import RewardConfig

RewardWeights = RewardConfig
DEFAULT_WEIGHTS = RewardWeights()
REWARD_COMPONENT_NAMES = ("money", "exp", "hp", "ep", "kill", "dead", "tower", "last_hit",
                          "economy_diff", "damage_dealt", "damage_taken", "nonhero_damage_taken")
ORGAN_TYPE_TOWER = 21
ORGAN_TYPE_CRYSTAL = 24
MONEY_UNIT = EXP_UNIT = 1000.0
ORGAN_UNIT = 12000.0
HP_UNIT_FALLBACK = 5000.0

def _enum_value(v) -> int:
    return v.value if hasattr(v, 'value') else v

def _num_of(obj, name: str, default=None):
    v = getattr(obj, name, None)
    if v is None:
        return default
    try:
        result = float(v)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default

def _camp_of(x, default=-1):
    c = getattr(x, 'camp', None)
    if c is None:
        return default
    try:
        return int(c)
    except (TypeError, ValueError):
        return int(getattr(c, 'value', default))

def _visible_to(req_pb, observer_camp, target_camp) -> Optional[bool]:
    if observer_camp is None or target_camp is None:
        return None
    for h in req_pb.hero_list:
        if _camp_of(h) == observer_camp:
            cv = getattr(h, 'camp_visible', None)
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

def extract_hero_snapshot(req_pb, camp, observer_camp: Optional[int]=None) -> Optional[Dict[str, float]]:
    for hero in req_pb.hero_list:
        if _camp_of(hero) != camp:
            continue
        enemy = next((h for h in req_pb.hero_list if _camp_of(h) != camp), None)
        hit_info = getattr(hero, 'hit_target_info', None)
        hit_records, skill_records = 0, 0
        for hit in hit_info if hit_info is not None else ():
            if enemy is not None and enemy.runtime_id in getattr(hit, 'hit_target', ()):
                hit_records += 1
                # 官方 SkillSlotType 的 1/2/3 为英雄技能；仅诊断，不再次发伤害奖。
                skill_records += int(int(hit.slot_type) in (1, 2, 3))
        return {
            'hp': _num_of(hero, 'hp', 0.0),
            'max_hp': _num_of(hero, 'max_hp', None),
            'money': _num_of(hero, 'moneyCnt', None),
            'wallet': _num_of(hero, 'money', None),
            'exp': _num_of(hero, 'exp', 0.0),
            'ep': _num_of(hero, 'ep', 0.0),
            'kills': _num_of(hero, 'killCnt', 0.0),
            'deaths': _num_of(hero, 'deadCnt', 0.0),
            'hurt_to_hero': _num_of(hero, 'totalHurtToHero', None),
            'hurt_by_hero': _num_of(hero, 'totalBeHurtByHero', None),
            'visible': _visible_to(req_pb, observer_camp, camp),
            'hit_info_available': hit_info is not None,
            'hero_hit_records': hit_records,
            'hero_skill_hit_records': skill_records,
        }
    return None

def find_organ_hp(req_pb, camp, organ_type: int) -> Optional[float]:
    for o in req_pb.organ_list:
        if o.camp != camp:
            continue
        if _enum_value(o.type) == organ_type:
            return float(o.hp)
    return None

def hp_term(prev: Dict, cur: Dict) -> float:
    """累计伤害计数器优先；不可见时 hp 可能是 1，不能盲目做血量差分。"""
    (my_p, my_c) = (prev['my'], cur['my'])
    (en_p, en_c) = (prev['en'], cur['en'])
    unit = my_c.get('max_hp') or HP_UNIT_FALLBACK
    if not unit or unit <= 0:
        unit = HP_UNIT_FALLBACK
    ctr_keys = ('hurt_to_hero', 'hurt_by_hero')
    if all((my_p.get(k) is not None for k in ctr_keys)) and all((my_c.get(k) is not None for k in ctr_keys)):
        dealt = max(0.0, my_c['hurt_to_hero'] - my_p['hurt_to_hero'])
        taken = max(0.0, my_c['hurt_by_hero'] - my_p['hurt_by_hero'])
        return (dealt - taken) / unit
    if not (my_p.get('visible') and my_c.get('visible') and en_p.get('visible') and en_c.get('visible')):
        return 0.0
    d_en = en_c['hp'] - en_p['hp']
    d_my = my_c['hp'] - my_p['hp']
    return (max(0.0, -d_en) - max(0.0, -d_my)) / unit

def compute_value_targets(rewards, gamma: float, values, last_value: float, lamda: float, dones=None) -> Tuple[np.ndarray, np.ndarray]:
    """GAE：δ=r+γV_next−V；A=δ+γλA_next。dones 仅标记真正终止。"""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    T = len(rewards)
    assert T == len(values), f'rewards/values 长度不一致：{T} vs {len(values)}'
    if dones is None:
        dones_arr = np.zeros(T, dtype=np.float32)
    else:
        dones_arr = np.asarray(dones, dtype=np.float32).reshape(-1)
        assert len(dones_arr) == T, f'dones 长度 {len(dones_arr)} != T {T}'
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones_arr[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lamda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return (advantages, returns)

def organ_snapshot(req_pb, camp):
    """分别跟踪塔和水晶，缺席返回 None；不猜测缺席的原因。"""
    return {kind: find_organ_hp(req_pb, camp, kind)
            for kind in (ORGAN_TYPE_TOWER, ORGAN_TYPE_CRYSTAL)}


def organ_damage(previous, current):
    # 只用前后两帧都存在的建筑，避免缺帧/重新出现产生幽灵奖励。
    return sum(max(0.0, previous[k] - current[k]) for k in previous
               if previous[k] is not None and current[k] is not None)


def counter_delta(previous, current, name):
    a, b = previous.get(name), current.get(name)
    return 0.0 if a is None or b is None else max(0.0, b - a)


def sdk_last_hit(state):
    """本地 SDK state.reward 为十维原始分量，索引 5 是有符号净补刀事件。

    返回值未乘 config.json 的 reward_last_hit；不要使用 env.step 返回的九维向量。
    正/负 credits 不能当成双方完整累计补刀数，零值也可能包含相互抵消。
    """
    raw = state.get("reward")
    if raw is None:
        return None, "missing"
    try:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (10,) or not np.isfinite(values).all():
            return None, "invalid"
        value = float(values[5])
        if abs(value - round(value)) > 1e-5:
            return None, "invalid"
        return float(round(value)), "available"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"


def monotonic_delta(previous, current, name):
    a, b = previous.get(name), current.get(name)
    if a is None or b is None or not np.isfinite([a, b]).all() or b < a:
        return None
    return b - a


def nonhero_damage_residual(previous, current, hero_damage):
    """存活相邻帧的血量下降扣除英雄伤害计数；未知/死亡跳变不猜测来源。"""
    mine_p, mine_c = previous["my"], current["my"]
    values = (mine_p.get("hp"), mine_c.get("hp"),
              mine_p.get("deaths"), mine_c.get("deaths"))
    if hero_damage is None or any(value is None for value in values):
        return None
    if not np.isfinite([*values, hero_damage]).all():
        return None
    hp_p, hp_c, deaths_p, deaths_c = values
    if hp_p <= 0 or hp_c <= 0 or deaths_c != deaths_p:
        return None
    # 同一步内治疗可能抵消部分伤害；这是保守可见量，不识别具体伤害来源。
    return max(0.0, hp_p - hp_c - hero_damage)


def economy_gap(snapshot):
    if snapshot is None:
        return None
    a, b = snapshot["my"].get("money"), snapshot["en"].get("money")
    return a - b if a is not None and b is not None and np.isfinite([a, b]).all() else None


def transition_events(prev, cur):
    """新奖励只用时间前进、计数器已知且未回绕的证据；None 与真实零分开。"""
    result = dict.fromkeys(("income_my", "income_enemy", "economy_diff_delta",
                           "damage_dealt", "damage_taken", "nonhero_damage_taken", "last_hit_net"))
    a, b = (prev or {}).get("frame_no"), (cur or {}).get("frame_no")
    if a is None or b is None or not np.isfinite([a, b]).all() or b <= a:
        return result
    mine_p, mine_c = prev["my"], cur["my"]
    result["income_my"] = monotonic_delta(mine_p, mine_c, "money")
    result["income_enemy"] = monotonic_delta(prev["en"], cur["en"], "money")
    if result["income_my"] is not None and result["income_enemy"] is not None:
        result["economy_diff_delta"] = result["income_my"] - result["income_enemy"]
    result["damage_dealt"] = monotonic_delta(mine_p, mine_c, "hurt_to_hero")
    result["damage_taken"] = monotonic_delta(mine_p, mine_c, "hurt_by_hero")
    result["nonhero_damage_taken"] = nonhero_damage_residual(prev, cur, result["damage_taken"])
    result["last_hit_net"] = cur.get("sdk_last_hit")
    return result


def normalized_damage(amount, maximum):
    return (amount / maximum if amount is not None and maximum is not None
            and np.isfinite(maximum) and maximum > 0 else 0.0)


def terminal_reward_terms(outcome, frame, initial_frame, max_frames, weights=DEFAULT_WEIGHTS):
    bonus = weights.win if outcome == "win" else weights.lose if outcome == "lose" else 0.0
    speed = 0.0
    if (outcome == "win" and frame is not None and initial_frame is not None
            and np.isfinite([frame, initial_frame]).all() and frame >= initial_frame and max_frames > 0):
        speed = weights.win_speed * max(0.0, 1 - (frame - initial_frame) / max_frames)
    return {"terminal": bonus, "win_speed": speed}


def state_reward_terms(prev, cur, weights=DEFAULT_WEIGHTS):
    my_p, my_c = prev["my"], cur["my"]
    damage = organ_damage(prev["en_organs"], cur["en_organs"])
    damage -= organ_damage(prev["my_organs"], cur["my_organs"])
    events = transition_events(prev, cur)
    return {
        "money": weights.money * counter_delta(my_p, my_c, "money") / MONEY_UNIT,
        "exp": weights.exp * counter_delta(my_p, my_c, "exp") / EXP_UNIT,
        "hp": weights.hp * hp_term(prev, cur),
        "ep": 0.0,
        "kill": weights.kill * counter_delta(my_p, my_c, "kills"),
        "dead": weights.dead * counter_delta(my_p, my_c, "deaths"),
        "tower": weights.tower * damage / ORGAN_UNIT,
        "last_hit": weights.last_hit * (events["last_hit_net"] or 0.0),
        "economy_diff": weights.economy_diff * (events["economy_diff_delta"] or 0.0) / MONEY_UNIT,
        "damage_dealt": weights.damage_dealt * normalized_damage(events["damage_dealt"], cur["en"].get("max_hp")),
        "damage_taken": weights.damage_taken * normalized_damage(events["damage_taken"], my_c.get("max_hp")),
        "nonhero_damage_taken": weights.nonhero_damage_taken * normalized_damage(
            events["nonhero_damage_taken"], my_c.get("max_hp")),
    }


def compute_state_reward(prev, cur, weights=DEFAULT_WEIGHTS):
    return float(sum(state_reward_terms(prev, cur, weights).values()))


def determine_outcome(req_pb, my_camp, en_camp):
    """仅凭明确的水晶摧毁证据判胜负；到时规则未核对的局保留为 unknown。"""
    mine = find_organ_hp(req_pb, my_camp, ORGAN_TYPE_CRYSTAL)
    enemy = find_organ_hp(req_pb, en_camp, ORGAN_TYPE_CRYSTAL)
    my_dead, en_dead = mine is not None and mine <= 0, enemy is not None and enemy <= 0
    if my_dead and en_dead:
        return "draw"
    if my_dead:
        return "lose"
    if en_dead:
        return "win"
    return None
