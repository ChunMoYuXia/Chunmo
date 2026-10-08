"""HoK 1v1 分层动作空间的 Action Mask 工具。

────────────────────────────────────────────────────────────────────────────
动作空间：6 个头，LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
  0: which_button (12)  —— 按钮选择（移动/攻击/技能/回城…）
  1: move_x (16)        —— 移动方向 x 分量
  2: move_z (16)        —— 移动方向 z 分量
  3: kill_x (16)        —— 技能方向 x 分量
  4: kill_z (16)        —— 技能方向 z 分量
  5: target (8)         —— 目标单位，**合法性依赖已选中的 button**

────────────────────────────────────────────────────────────────────────────
HoK 中存在两种 legal_action 格式，本模块负责两者的处理与换算：

  展开格式 (172-dim)：环境 state["legal_action"] 的原始格式
      12 + 16 + 16 + 16 + 16 + 96 = 172，其中最后 96 = 8 × 12，
      按 (button, target) 排布，即 reshape 成 (12, 8) 后第 i 行是
      "当 button=i 时，8 个 target 槽位的合法性"。
      来源：hok_env/hok/hok1v1/env1v1.py:263-266
            np.split(la, [12,28,44,60,76]) 后 tmp[-1].reshape(-1, 8)[button]

  压缩格式 (84-dim)：训练样本中存储的格式（官方 SERI_VEC_SPLIT_SHAPE 的第二段）
      12 + 16 + 16 + 16 + 16 + 8 = 84
      即把 (12, 8) 的 target mask 用"实际采样的 button"选中一行，
      压成与 button 无关的 8 维向量。
      来源：aiarena/1v1/actor/agent.py:246-253  _update_legal_action
      官方 Config: SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]

  ⚠️ 172 → 84 的换算**必须**在采样出 button 之后进行；
     这是原实现缺失的一环，会导致 target 头的 mask 完全错位。

────────────────────────────────────────────────────────────────────────────
数值实现严格对齐官方 ``_legal_soft_max``
（定义在 aiarena/1v1/actor/agent.py:479-491，**不在 env1v1.py**）：

    tmp  = logits - 1e20 * (1 - mask)
    tmp  = clip(tmp - max(tmp), -1e20, 1)
    tmp  = (exp(tmp) + 1e-5) * mask        # ← eps 在乘 mask 之前
    probs = tmp / sum(tmp)

关键点：eps 加在 **乘 mask 之前**，因此非法位置概率**严格为 0**。
原实现写成 ``mask * exp(tmp) + eps``（eps 在乘 mask 之后），
会给每个非法动作分配 1e-5 的概率质量，既与官方不一致，
也让 ``_smoke_test.py`` 里 ``illegal < 1e-6`` 的断言永远无法通过。
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 6 个动作头的维度
LABEL_SIZE_LIST: List[int] = [12, 16, 16, 16, 16, 8]

#: button 头维度（= LABEL_SIZE_LIST[0]），也等于 target mask 的行数
BUTTON_DIM: int = LABEL_SIZE_LIST[0]          # 12

#: target 头维度（= LABEL_SIZE_LIST[-1]），也等于 target mask 的列数
TARGET_DIM: int = LABEL_SIZE_LIST[-1]         # 8

#: 展开格式：前 5 个头 + 12×8 的二维 target mask
LEGAL_ACTION_SIZE_LIST_EXPANDED: List[int] = LABEL_SIZE_LIST[:-1] + [BUTTON_DIM * TARGET_DIM]
LEGAL_ACTION_TOTAL_EXPANDED: int = sum(LEGAL_ACTION_SIZE_LIST_EXPANDED)   # 172

#: 压缩格式：前 5 个头 + 1 个被 button 选中的 8 维 target mask
LEGAL_ACTION_SIZE_LIST_COMPRESSED: List[int] = list(LABEL_SIZE_LIST)
LEGAL_ACTION_TOTAL_COMPRESSED: int = sum(LEGAL_ACTION_SIZE_LIST_COMPRESSED)  # 84

#: 前 5 个头在两种格式中占据的前缀长度（76），换算时直接复用
_PREFIX_LEN: int = sum(LABEL_SIZE_LIST[:-1])  # 76

#: 数值稳定常数。1e20 在 float32 下安全（float32 上限约 3.4e38），
#: 但在 float16 下会溢出成 inf（float16 上限仅 65504），
#: 因此 masked_softmax 内部会先 .float()，见下方实现。
_MASK_NEG_INF: float = 1e20

#: 与官方 Config.MIN_POLICY = 1e-5 一致。
#: 注意不要用 1e-8：float16 下 1e-8 会被舍入为 0，等于没加。
_MASK_EPS: float = 1e-5

#: 全零 mask 的兜底策略：见 masked_softmax 的说明
_WARNED_ALL_ZERO_MASK = False


# ---------------------------------------------------------------------------
# 切分工具
# ---------------------------------------------------------------------------

def split_legal_action_expanded(legal_action):
    """把 172 维展开 legal_action 切分为 5 个一维 mask + 1 个 (12, 8) 二维 mask。

    与官方 env1v1.py:263-266 ``_split_legal_action`` 的切分方式一致。

    Args:
        legal_action: (..., 172) 的 tensor / ndarray
    Returns:
        list，长度 6：
          [0..4] 形状 (..., size_i)，size_i 依次为 12/16/16/16/16
          [5]    形状 (..., 12, 8)，第一维为 button 索引
    """
    la = torch.as_tensor(legal_action, dtype=torch.float32)
    splits = []
    start = 0
    for i, size in enumerate(LEGAL_ACTION_SIZE_LIST_EXPANDED):
        end = start + size
        if i == len(LEGAL_ACTION_SIZE_LIST_EXPANDED) - 1:
            # 最后一段是 target：还原成 (…, 12, 8)，行 = button，列 = target
            splits.append(la[..., start:end].reshape(*la.shape[:-1], BUTTON_DIM, TARGET_DIM))
        else:
            splits.append(la[..., start:end])
        start = end
    return splits


def split_legal_action_compressed(legal_action):
    """把 84 维压缩 legal_action 切分为 6 个一维 mask。

    Args:
        legal_action: (..., 84)
    Returns:
        list[Tensor]，形状依次为 [12],[16],[16],[16],[16],[8]
    """
    la = torch.as_tensor(legal_action, dtype=torch.float32)
    splits = []
    start = 0
    for size in LABEL_SIZE_LIST:
        splits.append(la[..., start:start + size])
        start += size
    return splits


# ---------------------------------------------------------------------------
# 172 → 84 换算（原实现缺失的关键环节）
# ---------------------------------------------------------------------------

def compress_legal_action(legal_action: torch.Tensor, button: torch.Tensor) -> torch.Tensor:
    """用已采样的 button 把 172 维展开 legal_action 压成 84 维。

    这是官方 ``Agent._update_legal_action``（aiarena/1v1/actor/agent.py:246-253）
    的批量版本：

        fix_part  = original_la[: -target_size * top_size]                 # 前 76 维
        target_la = original_la[-target_size*top_size:].reshape([top_size, target_size])[actions[0]]
        return concatenate([fix_part, target_la])                          # 76 + 8 = 84

    Args:
        legal_action: (..., 172) 展开格式的 legal_action
        button:       (...,) int 已采样的 button 索引，取值 [0, 12)
    Returns:
        (..., 84) 压缩格式。可直接存入训练样本 / 喂给 learner。
    """
    la = torch.as_tensor(legal_action, dtype=torch.float32)
    button = torch.as_tensor(button, dtype=torch.long)

    fix_part = la[..., :_PREFIX_LEN]                                   # (..., 76)
    target_2d = la[..., _PREFIX_LEN:].reshape(
        *la.shape[:-1], BUTTON_DIM, TARGET_DIM
    )                                                                  # (..., 12, 8)

    # 用 button 在 dim=-2 上做 gather，取出 (..., 1, 8) 再 squeeze 掉该维。
    # index 需要广播成 (..., 1, TARGET_DIM) 才能与 target_2d 对齐。
    index = button.unsqueeze(-1).unsqueeze(-1).expand(*button.shape, 1, TARGET_DIM)
    target_row = torch.gather(target_2d, -2, index).squeeze(-2)        # (..., 8)

    return torch.cat([fix_part, target_row], dim=-1)                   # (..., 84)


def get_target_mask_for_button(target_mask_2d: torch.Tensor,
                               button_idx: torch.Tensor) -> torch.Tensor:
    """从 (12, 8) 的 target mask 中取出指定 button 对应的 (8,) 行。

    与 ``compress_legal_action`` 取行逻辑相同，供采样阶段单独使用
    （原实现定义了本函数但从未调用，导致 target 采样用的是无条件的扁平 8 维 mask）。

    Args:
        target_mask_2d: (..., 12, 8)
        button_idx:     (...,) int
    Returns:
        (..., 8)
    """
    tm = torch.as_tensor(target_mask_2d, dtype=torch.float32)
    b = torch.as_tensor(button_idx, dtype=torch.long)
    index = b.unsqueeze(-1).unsqueeze(-1).expand(*b.shape, 1, TARGET_DIM)
    return torch.gather(tm, -2, index).squeeze(-2)


# ---------------------------------------------------------------------------
# Masked softmax
# ---------------------------------------------------------------------------

def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """对 logits 施加 legal action mask（非法位置 -> 极大负数）。

    仅用于需要"掩码后的原始 logit"的场合（例如调试 / 外部 softmax）。
    若随后要做 softmax 并取 log，请直接用 ``masked_softmax``，
    因为它额外处理了数值稳定与 eps 的施加顺序。

    Args:
        logits:     (..., dim)
        legal_mask: (..., dim)，取值 {0,1}
    Returns:
        与 logits 同形状
    """
    logits = logits.float()
    legal_mask = legal_mask.float()
    return logits - (1.0 - legal_mask) * _MASK_NEG_INF


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """带 legal action mask 的 softmax，语义与官方 ``_legal_soft_max`` 完全一致。

    官方实现（aiarena/1v1/actor/agent.py:479-491）::

        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)

    三处易错点，本实现逐一对应：
      1. **eps 在乘 mask 之前**：``(exp + eps) * mask`` 使非法位置严格为 0；
         写成 ``mask * exp + eps`` 会给非法动作残留 1e-5 量级的概率。
      2. **float32 计算**：1e20 在 float16 下溢出为 inf，而 AMP autocast 下
         logits 很可能是 float16，故先 ``.float()``。
      3. **全零 mask 兜底**：官方此处会得到 0/0 = NaN。本实现对全零 mask
         退化为"在 index 0 上取 one-hot"并告警一次，避免 NaN 传播；
         该 head 此时无合法动作，环境侧 ``_check_action`` 会给出 warning。

    Args:
        logits:     (..., dim)
        legal_mask: (..., dim)，取值 {0,1}
    Returns:
        (..., dim) 概率分布，非法位置为 0（全零 mask 情况除外）
    """
    global _WARNED_ALL_ZERO_MASK

    logits = logits.float()
    legal_mask = legal_mask.float()

    masked = logits - _MASK_NEG_INF * (1.0 - legal_mask)
    masked = masked - masked.max(dim=-1, keepdim=True).values
    masked = torch.clamp(masked, -_MASK_NEG_INF, 1.0)
    # 关键：先加 eps 再乘 mask（官方顺序）
    exp = (torch.exp(masked) + _MASK_EPS) * legal_mask

    denom = exp.sum(dim=-1, keepdim=True)
    all_zero = (denom <= 0).squeeze(-1)

    if bool(all_zero.any()):
        if not _WARNED_ALL_ZERO_MASK:
            _WARNED_ALL_ZERO_MASK = True
            import logging
            logging.getLogger(__name__).warning(
                "masked_softmax: 检测到全零 legal mask（该 head 无任何合法动作）。"
                "已退化为 index 0 的 one-hot 以避免 NaN；请检查上游 legal_action。"
            )
        # 用 one-hot(0) 替换 NaN 分支。denom 置 1 防止除零。
        safe_denom = torch.where(all_zero.unsqueeze(-1), torch.ones_like(denom), denom)
        probs = exp / safe_denom
        fallback = torch.zeros_like(probs)
        fallback[..., 0] = 1.0
        probs = torch.where(all_zero.unsqueeze(-1), fallback, probs)
        return probs

    return exp / denom


def probs_to_log_probs(probs: torch.Tensor) -> torch.Tensor:
    """概率 -> log 概率，用 ``_MASK_EPS`` 保证不会出现 -inf。

    官方在 ``_legal_soft_max`` 之后取 log 时用的也是 1e-5（Config.MIN_POLICY）。
    原实现用 1e-8，在 float16 下会被舍入为 0，等于没有保护。
    """
    return torch.log(probs + _MASK_EPS)


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def sample_from_probs(probs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    """从概率分布采样一个动作索引。

    Args:
        probs: (..., dim) 概率分布
        deterministic: True 时取 argmax（评估用）
    Returns:
        (...,) long tensor
    """
    if deterministic:
        return torch.argmax(probs, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1).squeeze(-1)
    return idx.reshape(probs.shape[:-1])


# ---------------------------------------------------------------------------
# 断言 / 调试
# ---------------------------------------------------------------------------

def assert_action_legal(action: Union[Sequence[int], np.ndarray, torch.Tensor],
                        legal_action: Union[Sequence[float], np.ndarray, torch.Tensor]) -> None:
    """运行时断言：6 维动作必须在 172 维展开 legal_action 下合法。

    作为防止 mask bug 的最后一道防线（前两道为网络 mask 与采样时校验）。
    仅在调试 / 单测中调用，训练热路径上不要用。

    Args:
        action:       6 个 int：(button, move_x, move_z, kill_x, kill_z, target)
        legal_action: (172,) 展开 legal_action
    """
    splits = split_legal_action_expanded(torch.as_tensor(legal_action, dtype=torch.float32))
    button = int(action[0])
    assert splits[0][button] == 1, f"非法 button: {button}"
    for i in range(1, 5):
        assert splits[i][int(action[i])] == 1, f"非法 head {i} 取值 {int(action[i])}"
    target_mask = splits[5][button]
    assert target_mask[int(action[5])] == 1, (
        f"非法 target {int(action[5])} (button={button})，"
        f"该 button 下的 target mask = {target_mask.tolist()}"
    )
