"""相邻观测的行为证据；记录变化，不把相关性写成命中或施法成功。"""
import math
from .reward import counter_delta, organ_damage

BEHAVIOR_KEYS = (
    "observed_frame_span", "snapshot_step_count", "income_gain", "enemy_income_gain",
    "hero_damage_gain", "hero_damage_taken", "enemy_building_damage", "own_building_damage",
    "move_attempts", "move_observed_displacements", "move_position_known_attempts")


def position_frame(state):
    req = state.get("req_pb")
    hero = next((h for h in getattr(req, "hero_list", [])
                 if h.runtime_id == state.get("player_id")), None)
    location = getattr(hero, "location", None)
    position = ((float(location.x), float(location.z))
                if location is not None and hasattr(location, "x") and hasattr(location, "z")
                and hero.hp > 0 else None)
    return getattr(req, "frame_no", None), position


def step_evidence(before_frame, after_frame, previous, current, button):
    result = dict.fromkeys(BEHAVIOR_KEYS, 0)
    frame, position = before_frame
    next_frame, next_position = after_frame
    if frame is not None and next_frame is not None:
        result["observed_frame_span"] = max(0, next_frame - frame)
    if button == 2:
        result["move_attempts"] = 1
        if position is not None and next_position is not None:
            result["move_position_known_attempts"] = 1
            result["move_observed_displacements"] = int(math.dist(position, next_position) > 1.0)
    if previous is not None and current is not None:
        result.update(snapshot_step_count=1,
            income_gain=counter_delta(previous["my"], current["my"], "money"),
            enemy_income_gain=counter_delta(previous["en"], current["en"], "money"),
            hero_damage_gain=counter_delta(previous["my"], current["my"], "hurt_to_hero"),
            hero_damage_taken=counter_delta(previous["my"], current["my"], "hurt_by_hero"),
            enemy_building_damage=organ_damage(previous["en_organs"], current["en_organs"]),
            own_building_damage=organ_damage(previous["my_organs"], current["my_organs"]))
    return result


def summarize_behavior(games):
    """旧日志缺失证据时返回空对象；完整局或同批片段均可按观察量求和。"""
    keys = (*BEHAVIOR_KEYS, "attack_attempts", "attack_without_enemy_target",
            "attack_target_constraint_decisions", "attack_button_blocked_decisions")
    rows = [g.get("state_diagnostics", {}) for g in games]
    if not any("snapshot_step_count" in row for row in rows):
        return {}
    counts = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    rates = {}
    for name, numerator, denominator in (
        ("attack_without_enemy_target_rate", "attack_without_enemy_target", "attack_attempts"),
        ("move_observed_displacement_rate", "move_observed_displacements", "move_position_known_attempts")):
        rates[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
    for key in ("income_gain", "enemy_income_gain", "hero_damage_gain", "enemy_building_damage"):
        rates[key + "_per_1000_frames"] = (counts[key] * 1000 / counts["observed_frame_span"]
                                          if counts["observed_frame_span"] else None)
    return {**counts, **rates}
