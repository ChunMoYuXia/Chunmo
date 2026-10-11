"""奖励证据与覆盖率：未检测到事件和检测字段缺失分别记录。"""
from .reward import economy_gap, transition_events


REWARD_DETECTION_KEYS = (
    "observed_steps", "economy_available_steps", "economy_diff_delta",
    "damage_dealt_available_steps", "damage_taken_available_steps",
    "damage_dealt_steps", "damage_taken_steps", "damage_dealt_raw", "damage_taken_raw",
    "nonhero_damage_taken_available_steps", "nonhero_damage_taken_steps", "nonhero_damage_taken_raw",
    "last_hit_available_steps", "last_hit_missing_steps", "last_hit_invalid_steps",
    "last_hit_net", "last_hit_positive_credits", "last_hit_negative_credits",
    "hit_info_available_steps", "hero_skill_hit_records", "enemy_skill_hit_records",
    "attack_followed_by_damage_steps", "skill_followed_by_damage_steps",
    "attack_followed_by_last_hit_steps", "skill_followed_by_last_hit_steps")


def step_reward_detection(previous, current, button):
    result = dict.fromkeys(REWARD_DETECTION_KEYS, 0)
    result["observed_steps"] = 1
    events = transition_events(previous, current)
    for event, key in (("economy_diff_delta", "economy"), ("damage_dealt", "damage_dealt"),
                       ("damage_taken", "damage_taken"), ("last_hit_net", "last_hit")):
        result[key + "_available_steps"] = int(events[event] is not None)
    result["economy_diff_delta"] = events["economy_diff_delta"] or 0.0
    for key in ("damage_dealt", "damage_taken"):
        result[key + "_raw"] = events[key] or 0.0
        result[key + "_steps"] = int((events[key] or 0) > 0)
    residual = events["nonhero_damage_taken"]
    result["nonhero_damage_taken_available_steps"] = int(residual is not None)
    result["nonhero_damage_taken_raw"] = residual or 0.0
    result["nonhero_damage_taken_steps"] = int((residual or 0) > 0)
    net = events["last_hit_net"]
    result["last_hit_net"] = net or 0.0
    result["last_hit_positive_credits"] = max(0, net or 0)
    result["last_hit_negative_credits"] = max(0, -(net or 0))
    if net is None:
        status = (current or {}).get("sdk_last_hit_status", "missing")
        result["last_hit_invalid_steps" if status == "invalid" else "last_hit_missing_steps"] = 1
    if current:
        mine, enemy = current["my"], current["en"]
        result["hit_info_available_steps"] = int(
            mine.get("hit_info_available", False) and enemy.get("hit_info_available", False))
        result["hero_skill_hit_records"] = mine.get("hero_skill_hit_records", 0)
        result["enemy_skill_hit_records"] = enemy.get("hero_skill_hit_records", 0)
    # 只是按钮之后观测到结果，不据此断言命中来自该按钮。
    for name, active in (("attack", button == 3), ("skill", button in (4, 5, 6, 10))):
        result[name + "_followed_by_damage_steps"] = int(active and (events["damage_dealt"] or 0) > 0)
        result[name + "_followed_by_last_hit_steps"] = int(active and (net or 0) > 0)
    return result


def summarize_reward_detection(games):
    rows = [g["reward_detection"] for g in games if "reward_detection" in g]
    if not rows:
        return {}
    counts = {key: sum(row.get(key, 0) for row in rows) for key in REWARD_DETECTION_KEYS}
    for name in ("economy", "damage_dealt", "damage_taken", "nonhero_damage_taken", "last_hit", "hit_info"):
        counts[name + "_coverage"] = (counts[name + "_available_steps"] / counts["observed_steps"]
                                      if counts["observed_steps"] else None)
    gaps = [economy_gap(g.get("final_snapshot")) for g in games]
    gaps = [gap for gap in gaps if gap is not None]
    counts["snapshot_economy_gap_mean"] = sum(gaps) / len(gaps) if gaps else None
    return counts
