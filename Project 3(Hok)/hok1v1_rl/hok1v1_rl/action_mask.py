"""合法动作工具：六头动作，target 的合法性取决于先选出的 button。

SDK 的 172 维 = 前五头 76 维 + 12×8 目标表；训练仅保存已选行，共 84 维。
"""
import logging
import numpy as np
import torch

LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
BUTTON_DIM, TARGET_DIM = 12, 8
LEGAL_ACTION_SIZE_LIST_EXPANDED = LABEL_SIZE_LIST[:-1] + [96]
LEGAL_ACTION_SIZE_LIST_COMPRESSED = LABEL_SIZE_LIST.copy()
LEGAL_ACTION_TOTAL_EXPANDED = 172
LEGAL_ACTION_TOTAL_COMPRESSED = 84
_PREFIX_LEN = 76
_MASK_NEG_INF = 1e20
_MASK_EPS = 1e-5
_WARNED_ALL_ZERO_MASK = False


def _position_xz(actor):
    location = getattr(actor, "location", None)
    if location is None or not hasattr(location, "x") or not hasattr(location, "z"):
        return None
    x, z = float(location.x), float(location.z)
    if not np.isfinite([x, z]).all() or max(abs(x), abs(z)) >= 90000:
        return None
    return x, z


def _guard_uncovered_tower_approach(legal, req, hero, radius):
    """塔前没有更靠近敌塔的己方兵线时，限制朝敌方半场的移动分量。"""
    if radius <= 0 or hero is None or float(hero.hp) <= 0 or legal[2] != 1:
        return
    hero_xy = _position_xz(hero)
    if hero_xy is None:
        return
    enemy_towers = [tower for tower in getattr(req, "organ_list", [])
                    if int(tower.type) == 21 and int(tower.camp) != int(hero.camp)
                    and float(tower.hp) > 0 and _position_xz(tower) is not None]
    if not enemy_towers:
        return
    tower = min(enemy_towers, key=lambda item: sum((a-b)**2 for a, b in zip(hero_xy, _position_xz(item))))
    tower_xy = _position_xz(tower)
    hero_distance2 = sum((a-b)**2 for a, b in zip(hero_xy, tower_xy))
    camp = int(hero.camp)
    if camp not in (1, 2):
        return
    forward = 1 if camp == 1 else -1
    axis_to_tower = [forward * (tower_xy[i] - hero_xy[i]) for i in (0, 1)]
    if hero_distance2 > radius**2 and all(value > 0 for value in axis_to_tower):
        return
    # 只有比我方英雄更靠近敌塔的活兵才算掩护；信息不全时不猜测兵线。
    soldiers = getattr(req, "soldier_list", None)
    if soldiers is None:
        return
    covered = False
    for soldier in soldiers:
        if int(soldier.camp) != int(hero.camp) or float(soldier.hp) <= 0:
            continue
        soldier_xy = _position_xz(soldier)
        if soldier_xy is None:
            continue
        soldier_distance2 = sum((a-b)**2 for a, b in zip(soldier_xy, tower_xy))
        if soldier_distance2 < hero_distance2 and soldier_distance2 <= (0.75*radius)**2:
            covered = True
            break
    # 旧 trace 的红蓝双方实测：移动方向头 1/2 的索引 8..15 不可靠；
    # 特别是 8/8 会发出 MoveDir -90，红方继续向敌塔后方移动，而非停步。
    # 只缩小 SDK 合法集合；无后退候选时保留原掩码，避免全零头。
    for axis, offset in enumerate((12, 28)):
        # 有兵线时仍不允许越过存活的一塔；留 1000 单位余量抵消单步位移。
        if covered and axis_to_tower[axis] > 1000:
            continue
        head = legal[offset:offset+16]
        if np.any(head[1:8]) and np.any(head[8:16]):
            head[8:16] = 0


def effective_legal_action(state, action_config):
    """先修改掩码再采样，保存同一掩码供 learner 重算；绝不事后替换动作。

    targeted_attack_v1 是普通射手的领域约束候选，不表示 SDK 非法动作。
    目标槽位：0 空、1 敌方英雄、2 自己、3..6 小兵、7 建筑。
    不扩大 SDK 合法集合；信息不足、target 头不生效或没有替代按钮时保留原规则。
    """
    raw = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)
    if raw.shape != (172,):
        raise ValueError(f"SDK legal_action 应为 172 维，实际 {raw.shape}")
    legal = raw.copy()
    if action_config.rule == "sdk":
        return legal
    if action_config.rule != "targeted_attack_v1":
        raise ValueError(f"未知动作规则：{action_config.rule}")
    req = state.get("req_pb")
    hero = next((h for h in getattr(req, "hero_list", [])
                 if h.runtime_id == state.get("player_id")), None)
    if hero is not None and int(hero.config_id) in action_config.targeted_attack_hero_ids:
        masks = state.get("sub_action_mask")
        if isinstance(masks, dict):
            row = masks.get(3, masks.get("3"))
        else:
            row = masks[3] if masks is not None and len(masks) > 3 else None
        if row is not None and len(row) == 6 and row[5] == 1 and legal[3] == 1:
            target = legal[76:].reshape(12, 8)[3]
            if target[[1, 3, 4, 5, 6, 7]].any():
                target[[0, 2]] = 0
            elif np.delete(legal[:12], 3).any():
                legal[3] = 0
    _guard_uncovered_tower_approach(legal, req, hero, action_config.tower_guard_radius)
    return legal


def split_legal_action_expanded(legal_action):
    legal = torch.as_tensor(legal_action, dtype=torch.float32)
    if legal.shape[-1] != 172:
        raise ValueError("环境 legal_action 必须为 172 维")
    parts = list(torch.split(legal, LEGAL_ACTION_SIZE_LIST_EXPANDED, dim=-1))
    parts[-1] = parts[-1].reshape(*legal.shape[:-1], 12, 8)
    return parts


def split_legal_action_compressed(legal_action):
    legal = torch.as_tensor(legal_action, dtype=torch.float32)
    if legal.shape[-1] != 84:
        raise ValueError("训练 legal_action 必须为 84 维")
    return list(torch.split(legal, LABEL_SIZE_LIST, dim=-1))


def get_target_mask_for_button(target_mask_2d, button_idx):
    target = torch.as_tensor(target_mask_2d, dtype=torch.float32)
    button = torch.as_tensor(button_idx, dtype=torch.long, device=target.device)
    index = button[..., None, None].expand(*button.shape, 1, TARGET_DIM)
    return target.gather(-2, index).squeeze(-2)


def compress_legal_action(legal_action, button):
    legal = torch.as_tensor(legal_action, dtype=torch.float32)
    targets = split_legal_action_expanded(legal)[-1]
    target_row = get_target_mask_for_button(targets, button)
    return torch.cat([legal[..., :76], target_row], dim=-1)


def mask_logits(logits, legal_mask):
    return logits.float() - (1 - legal_mask.float()) * _MASK_NEG_INF


def masked_softmax(logits, legal_mask):
    """非法概率严格为零；统一 float32，采样与训练使用完全相同的公式。"""
    global _WARNED_ALL_ZERO_MASK
    legal = legal_mask.float()
    masked = mask_logits(logits, legal)
    shifted = masked - masked.max(dim=-1, keepdim=True).values
    # epsilon 在 mask 乘法之前，不能给非法动作留下小概率。
    numerator = (shifted.clamp(-_MASK_NEG_INF, 1).exp() + _MASK_EPS) * legal
    denominator = numerator.sum(dim=-1, keepdim=True)
    empty = denominator <= 0
    probs = numerator / torch.where(empty, torch.ones_like(denominator), denominator)
    if bool(empty.any()):
        if not _WARNED_ALL_ZERO_MASK:
            logging.getLogger(__name__).warning("动作头 mask 全零，临时回退到动作 0；请检查 SDK 掩码")
            _WARNED_ALL_ZERO_MASK = True
        fallback = torch.zeros_like(probs)
        fallback[..., 0] = 1
        probs = torch.where(empty, fallback, probs)
    return probs


def probs_to_log_probs(probs):
    return torch.log(probs + _MASK_EPS)


def sample_from_probs(probs, deterministic=False):
    if deterministic:
        return probs.argmax(dim=-1)
    chosen = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).squeeze(-1)
    return chosen.reshape(probs.shape[:-1])


def assert_action_legal(action, legal_action):
    """诊断时逐项检查；target 必须使用该 button 对应的行。"""
    masks = split_legal_action_expanded(legal_action)
    button = int(action[0])
    for head in range(5):
        assert masks[head][int(action[head])] == 1, f"非法动作：head={head}, action={action[head]}"
    assert masks[-1][button, int(action[-1])] == 1, "非法 target"
