"""PPO Learner —— AMP + Grad Clip + KL 双阈值 + 逐头 ratio + Multi-head value loss。

────────────────────────────────────────────────────────────────────────────
⚠️ 关于"官方是联合 ratio"这个说法：**是错的，已纠正**
────────────────────────────────────────────────────────────────────────────
旧版本文件（以及 ``action_mask``/报告）声称官方把 6 个 head 的 log-prob
增量累加成一个标量、再取一次 exp 得到"单一联合 ratio"，并据此实现。核对
官方 ``aiarena/1v1/common/algorithm_torch.py::Algorithm.compute_loss`` 后
可以确认**恰恰相反**：``final_log_p`` 是在 ``for task_index`` 循环**体内**
被重新置零的，``ratio = torch.exp(final_log_p)`` 也在循环体内::

    for task_index in range(len(self.is_reinforce_task_list)):
        if self.is_reinforce_task_list[task_index]:
            final_log_p = torch.tensor(0.0)              # ← 每个 head 都重置
            ...
            final_log_p = final_log_p + policy_log_p - old_policy_log_p
            ratio      = torch.exp(final_log_p)          # ← 因此是**单头** ratio
            surr1      = ratio.clamp(0.0, 3.0) * advantage
            surr2      = ratio.clamp(1-c, 1+c) * advantage
            temp_policy_loss = -torch.sum(
                torch.minimum(surr1, surr2) * weight_list[task_index]
            ) / torch.maximum(torch.sum(weight_list[task_index]), 1.0)
            self.policy_cost = self.policy_cost + temp_policy_loss

所以官方是：**每个 head 各自 ratio、各自 clip、再按 head 权重加权求和**。
其中 ``weight_list[i]`` 就是样本里的 ``weight0..5``
（= ``sub_action_mask[选中的 button]``），用来屏蔽与该 button 无关的头。

原实现的两个偏差（本次均已修正，见 ``compute_loss``）：
  1. 把 6 头 log 比累加求 exp，得到"乘积 ratio"。乘积 ratio 的 clip 等价于
     把每头容差收紧到 ≈0.2/√6≈0.08，且任一头变差就整条样本梯度消失，
     策略更新近乎停滞。
  2. 完全没有 head 屏蔽（``sub_action_mask``），无关头也进入 ratio 与熵，
     把 KL 抬高、更容易撞上早停阈值。

────────────────────────────────────────────────────────────────────────────
其余改动（对齐官方 / 修历史遗留）
────────────────────────────────────────────────────────────────────────────
2. **6 个头全部参与，包含 target 头，并按下发的 head 权重屏蔽无关头**
   原实现显式跳过 head 5（``# TODO target head loss``），
   导致 target 头只被采样、从不被优化。

3. **真正的 minibatch**
   原实现在同一个完整 batch 上重复 epoch 次数，``minibatch_size`` 从未被使用。
   现按 **序列维** 切 minibatch（保持 LSTM 的时间结构），每个 epoch 重新打乱。

4. **数值安全**
   ``action_mask.masked_softmax`` 内部统一转 float32，避免 1e20 在 fp16 下溢出；
   log 前的 eps 用 1e-5（官方 MIN_POLICY），而不是在 fp16 下会被舍入为 0 的 1e-8。

5. **双 KL 阈值真正生效**
   ``kl_max`` 硬阈值：立即终止整个 update；
   ``kl_target`` 软阈值：跳过本 epoch 剩余的 minibatch。
   原实现只用了 kl_max，kl_target 是死配置。

6. **熵系数退火真正生效**
   ``entropy_beta`` 在训练过程中从 ``entropy_beta_start`` 线性退火到
   ``entropy_beta_end``（原实现只用 start，end 是死配置）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import math

import numpy as np
import torch
import torch.nn as nn

from .action_mask import LABEL_SIZE_LIST, masked_softmax, _MASK_EPS
from .reward import RunningRewardNormalizer
from .vram_guard import VRamGuard

logger = logging.getLogger(__name__)

#: 动作头数量（6：button / move_x / move_z / kill_x / kill_z / target）
NUM_ACTION_HEADS = len(LABEL_SIZE_LIST)

#: 每个 head 在 84 维压缩 legal_action 中的切片上下界。
#: 因为 LABEL_SIZE_LIST = [12,16,16,16,16,8]，前缀和恰好是
#: [(0,12), (12,28), (28,44), (44,60), (60,76), (76,84)]
_HEAD_SLICES: List[Tuple[int, int]] = []
_start = 0
for _size in LABEL_SIZE_LIST:
    _HEAD_SLICES.append((_start, _start + _size))
    _start += _size


def _make_amp_objects(enabled: bool):
    """兼容地构造 (autocast_cm_factory, GradScaler)。

    不同 torch 版本对 AMP API 的路径不同：
      * torch >= 2.4 推荐 ``torch.amp.autocast("cuda", ...)`` /
        ``torch.amp.GradScaler("cuda", ...)``
      * 更早版本只有 ``torch.cuda.amp.autocast`` / ``torch.cuda.amp.GradScaler``
    这里做一次能力探测，避免因版本差异直接崩。
    """
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=enabled)

        def autocast():
            """返回 AMP 上下文管理器（新版 torch.amp 路径）。"""
            return torch.amp.autocast("cuda", enabled=enabled)

        return autocast, scaler
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)

        def autocast():
            """返回 AMP 上下文管理器（旧版 torch.cuda.amp 路径）。"""
            return torch.cuda.amp.autocast(enabled=enabled)

        return autocast, scaler


@dataclass
class PPOStepResult:
    """单次 PPO 更新的结果汇总。"""
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    grad_norm: float
    lr: float
    clip_fraction: float
    explained_variance: float
    entropy_beta: float
    num_minibatches: int
    early_stopped: bool
    #: 本 batch 有效决策占比（1.0 = 没有尾部补齐）。它是 valid_mask 真的
    #: 透传到 loss 的证据；若恒为 1.0 而 env/chunk_padded_steps > 0，
    #: 说明 valid_mask 在 batch 组装时被丢了。
    valid_ratio: float = 1.0
    #: 采样侧与训练侧 old log prob 的最大平均绝对差（本 update 第一次前向、
    #: 参数未更新时测得）。正常 ≈ 1e-5，> OLD_NEW_GAP_TOLERANCE(0.05) 说明
    #: 两侧不是同一个前向函数 —— 此时 ratio/KL/clip 全部不可信。
    old_new_logp_gap: float = 0.0


class PPOLearner:
    """单 GPU PPO learner，集成 AMP + KL 双阈值 + 逐头 ratio + Multi-head value。"""

    def __init__(self, network, cfg, logger_obj, vram_guard: VRamGuard):
        self.network = network
        self.cfg = cfg
        self.ppo_cfg = cfg.ppo
        self.logger = logger_obj
        self.vram_guard = vram_guard
        self.device = next(network.parameters()).device

        # Adam：eps=1e-5 比 torch 默认 1e-8 大，配合 AMP 更稳
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=self.ppo_cfg.lr, eps=1e-5
        )
        self._autocast, self.scaler = _make_amp_objects(enabled=cfg.train.amp)

        # 回报在线归一化器（Welford）：跨 batch 维护 returns 的运行均值/方差，
        # value 回归目标始终是全局一致的零均值单位方差分布，替代原先的
        # 逐 batch z-score（后者使回归目标尺度随 batch 漂移，vloss 不稳、
        # 梯度爆炸的根源之一）。仅在 cfg.ppo.normalize_returns 时使用。
        self.return_normalizer = RunningRewardNormalizer()

        self._total_updates = 0
        self._lr_schedule = None
        self._lr_total_steps = None   # 用于熵系数退火的进度基准
        self._warned_missing_head_weights = False
        self._logged_head_weights = False
        #: 上一次 _report_old_new_gap 测到的 gap，供 update() 放进结果里
        self._last_old_new_gap = 0.0

    # -- 调度 -------------------------------------------------------------

    def set_lr_schedule(self, total_steps: int):
        """在 total_steps 内将 LR 从 cfg.lr 线性衰减到 cfg.lr_end。

        同时记录训练总步数，供 ``entropy_beta_at`` 计算退火进度。
        """
        from torch.optim.lr_scheduler import LambdaLR

        start_lr = self.ppo_cfg.lr
        end_lr = self.ppo_cfg.lr_end
        ratio_end = end_lr / start_lr if start_lr > 0 else 0.0

        self._lr_total_steps = max(1, total_steps)
        self._lr_schedule = LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: max(0.0, 1.0 - s / self._lr_total_steps)
            * (1.0 - ratio_end) + ratio_end,
        )

    def entropy_beta_at(self) -> float:
        """当前熵系数（在总步数上从 start 线性退火到 end）。

        前期强探索、后期收敛利用。这是 ``entropy_beta_end`` 真正生效的地方。
        """
        start = self.ppo_cfg.entropy_beta_start
        end = self.ppo_cfg.entropy_beta_end
        if not self._lr_total_steps:
            return start
        progress = min(1.0, self._total_updates / self._lr_total_steps)
        return start + (end - start) * progress

    # -- 损失 -------------------------------------------------------------

    def compute_loss(
        self,
        features: torch.Tensor,           # (B, T, 725)
        legal_actions: torch.Tensor,      # (B, T, 84) 已按 button 压缩
        actions: torch.Tensor,            # (B, T, 6)  long
        old_log_probs: torch.Tensor,      # (B, T, 6)  旧策略下的 log 概率
        advantages: torch.Tensor,         # (B, T)
        returns: torch.Tensor,            # (B, T)
        old_values: torch.Tensor,         # (B, T)
        head_weights: Optional[torch.Tensor] = None,   # (B, T, 6) sub_action_mask[button]
        valid_mask: Optional[torch.Tensor] = None,     # (B, T) 1=真实决策 0=尾部补齐
        lstm_state_init=None,
        entropy_beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算 PPO 损失（policy + value - entropy）及诊断量。

        返回 tensor dict（未 backward），由 ``update`` 负责反向。

        Args:
            head_weights: (B, T, 6)，官方样本里的 ``weight0..5``，即
                选中该 button 后各动作头是否生效（``sub_action_mask[button]``）。
                None 表示不做屏蔽（全 1）。
            valid_mask: (B, T)，``env_runner`` 为了让终局步进入 batch 而对
                尾部做的**零补齐**标记（1=真实决策，0=补齐）。None 表示全有效。
                ⚠️ 它必须参与**归一化**而不只是乘上去：本函数的
                policy_loss / entropy / clip_fraction 都是 ``Σw`` 归一化的商，
                若把补齐的 0 也算进分母，真实样本的梯度会被稀释。
        """
        if entropy_beta is None:
            entropy_beta = self.entropy_beta_at()

        # ---- 前向（AMP autocast）----
        with self._autocast():
            out = self.network(features, lstm_state_init)
            logits_list = out["logits_list"]
            new_values = out["value"]            # (B, T)
            value_heads = out["value_heads"]     # 目前仅用于日志

        # ---- 逐 head 计算 masked 概率与 log 概率 ----
        # 注意：masked_softmax 内部转 float32，因此下面所有概率/对数运算
        # 都在 float32 下进行，不受 autocast 的 fp16 影响（见 C7）。
        new_logp_per_head: List[torch.Tensor] = []
        probs_per_head: List[torch.Tensor] = []

        for i in range(NUM_ACTION_HEADS):
            logits_i = logits_list[i]                       # (B, T, size_i)
            s, e = _HEAD_SLICES[i]
            la_i = legal_actions[..., s:e]                  # (B, T, size_i)
            probs_i = masked_softmax(logits_i, la_i)        # (B, T, size_i)
            a_i = actions[..., i:i + 1]                     # (B, T, 1)
            p_taken = probs_i.gather(-1, a_i).squeeze(-1)   # (B, T)
            new_logp_per_head.append(torch.log(p_taken + _MASK_EPS))
            probs_per_head.append(probs_i)

        # ---- 逐头 log ratio ----
        per_head_log_ratios: List[torch.Tensor] = []
        for i in range(NUM_ACTION_HEADS):
            per_head_log_ratios.append(new_logp_per_head[i] - old_log_probs[..., i])

        # ---- valid_mask 归一化因子 ----
        # 尾部补齐的 0 必须从所有"Σw 归一化"的分母里剔除，否则真实样本的
        # 梯度会被按比例稀释（详见签名处的说明）。
        if valid_mask is None:
            valid = torch.ones_like(advantages, dtype=torch.float32)
        else:
            valid = valid_mask.to(device=advantages.device, dtype=torch.float32)
        valid_cnt = valid.sum().clamp(min=1.0)
        #: 把 padded 位置的逐头 log ratio 归零：使那里的 ratio 恒为 1、
        #: KL≈0、clip 不命中，从而对损失与诊断量都不产生任何影响。
        per_head_log_ratios = [
            d * valid for d in per_head_log_ratios
        ]

        # ---- 自检量：old/new log prob 的每头平均绝对差 ----
        # 权重未更新时它必须 ≈ 0。若明显 > 0，说明**采样侧与训练侧跑的不是
        # 同一个函数**（历史上就是时间维 attention / dropout / AMP 精度造成的），
        # 此时 ratio、KL、value_clip 全部失去意义，PPO 学不动。
        old_new_logp_gap = torch.stack(
            [d.abs().mean() for d in per_head_log_ratios]
        ).mean()

        # ---- head 权重（官方 sample 里的 weight0..5 = sub_action_mask[button]）----
        # 它表示"选了该 button 之后，这 6 个头里哪些真正生效"。
        # 缺失时退化为全 1（等价于不做屏蔽）。
        if head_weights is None:
            head_weights = torch.ones(
                *advantages.shape, NUM_ACTION_HEADS,
                device=advantages.device, dtype=torch.float32,
            )
        else:
            head_weights = head_weights.to(
                device=advantages.device, dtype=torch.float32
            )

        # ---- PPO clipped surrogate：**逐头**独立 clip（严格对齐官方）----
        # 官方 algorithm_torch.compute_loss 在 `for task_index` 循环体内把
        # final_log_p **重新置零**，所以 ratio = exp(final_log_p) 是**单头** ratio：
        #     surr1 = ratio.clamp(0, 3) * advantage
        #     surr2 = ratio.clamp(1-c, 1+c) * advantage
        #     loss_i = -Σ min(surr1, surr2)·w_i / max(Σ w_i, 1)
        #
        # ⚠️ 这里曾被写成"6 头 log 比累加后取一次 exp"（乘积 ratio），有两个后果：
        #    ① clip 作用在乘积上 ⇒ 每头容差被收紧到 ≈0.2/√6≈0.08；
        #    ② 任一头变差整条样本就被 clip 掉 ⇒ 梯度消失、更新近乎停滞。
        # ⚠️ w_i 是官方的 `weight0..5`（= sub_action_mask[选中 button]），用来屏蔽
        #    "与该 button 无关的头"。缺了它，选了"移动"时技能/目标头也进 ratio 与熵，
        #    噪声被放大、KL 更容易撞早停阈值。
        policy_loss = torch.zeros((), device=new_values.device, dtype=torch.float32)
        total_entropy = torch.zeros((), device=new_values.device, dtype=torch.float32)
        clip_hits = torch.zeros((), device=new_values.device, dtype=torch.float32)

        # 形状自检（永久保留）。必须**显式**报出每个张量的形状：
        # 只报 "tensor a (512) vs b (16)" 时无法看出是谁错了，而这类错误
        # 在真实训练里排查代价很高 —— 曾经就是因为 advantages 被多补了
        # 一个尾维 (B,T,1)，与 ratio_i (B,T) 广播成 (B,B,T) 才炸的。
        _b, _t = advantages.shape[0], advantages.shape[1]
        for _name, _tensor, _want in (
            ("returns", returns, (_b, _t)),
            ("old_values", old_values, (_b, _t)),
            ("new_values", new_values, (_b, _t)),
            ("valid_mask", valid, (_b, _t)),
            ("head_weights", head_weights, (_b, _t, NUM_ACTION_HEADS)),
            ("actions", actions, (_b, _t, NUM_ACTION_HEADS)),
        ):
            _actual = tuple(_tensor.shape)
            if _actual != _want:
                raise RuntimeError(
                    "compute_loss 形状不匹配：%s 实际 %s，期望 %s。"
                    "（提示：一维量被多补了尾维会变成 (B,T,1)，"
                    "与 (B,T) 逐元素相乘时会广播成 (B,B,T)。）"
                    % (_name, _actual, _want)
                )

        for i in range(NUM_ACTION_HEADS):
            ratio_i = torch.exp(per_head_log_ratios[i])
            surr1_i = torch.clamp(
                ratio_i,
                self.ppo_cfg.ratio_clamp_low, self.ppo_cfg.ratio_clamp_high,
            ) * advantages
            surr2_i = torch.clamp(
                ratio_i, 1.0 - self.ppo_cfg.clip_param,
                1.0 + self.ppo_cfg.clip_param,
            ) * advantages

            w_i = head_weights[..., i]                       # (B, T)
            # P0 的值恒为 0，所以 padded 位置对下面的 Σ 贡献恰好为 0；
            # 分母用同一个幂等掩码归一化，保证真实样本的梯度不被稀释。
            w_i = w_i * valid
            w_sum = w_i.sum().clamp(min=1.0)                 # 官方 max(Σw, 1)

            policy_loss = policy_loss - (
                torch.min(surr1_i, surr2_i) * w_i
            ).sum() / w_sum
            clip_hits = clip_hits + (
                ((ratio_i - 1.0).abs() > self.ppo_cfg.clip_param).float() * w_i
            ).sum() / w_sum

            # 熵：只在合法动作上统计（非法位置概率为 0），并按 head 权重加权
            s, e = _HEAD_SLICES[i]
            mask_i = legal_actions[..., s:e].float()
            p = probs_per_head[i]
            ent_i = -(p * torch.log(p + _MASK_EPS) * mask_i).sum(-1)   # (B, T)
            total_entropy = total_entropy + (ent_i * w_i).sum() / w_sum

        # 官方对 6 个头是**求和**；这里保留"按头平均"，以维持 value_coef /
        # entropy_beta 既有的相对量级（Adam 对全局缩放不敏感，但
        # policy : value 的相对权重会变）。若要逐位对齐官方，
        # 去掉这两行除法并把 PPOConfig.value_coef 改成 1.0。
        policy_loss = policy_loss / NUM_ACTION_HEADS
        total_entropy = total_entropy / NUM_ACTION_HEADS
        clip_fraction = clip_hits / NUM_ACTION_HEADS

        # ---- KL：按头平均的近似 KL（恒 >= 0），用于早停判据 ----
        # Schulman 估计器 KL_i ≈ exp(d_i) - 1 - d_i（对任意实数 d 恒 >= 0；
        # 出现负值只可能是 NaN 污染）。这里的 clamp(min=0) 只是双保险。
        #
        # ⚠️ 必须**逐头**度量再平均，不能直接用 6 头联合比率 exp(Σd_i)：
        #    后者的近似 KL 约为单头的 6 倍，配 0.015/0.05 的阈值时几乎每个
        #    update 的第一个 minibatch 就硬停（实测 25 步里 18 步早停、
        #    minibatch 只用到 1/8，策略几乎不更新）。逐头平均后与阈值量纲对齐
        #    （OpenAI 单动作策略惯例 target 0.015~0.02），且与上面逐头 clip 口径一致。
        kl = torch.stack([
            (torch.exp(d_i) - 1.0 - d_i).mean()
            for d_i in per_head_log_ratios
        ]).mean().clamp(min=0.0)

        # clip_fraction 已在上面逐头累加（见 clip_hits）。

        # ---- Value loss ----
        # 用 masked mean：padded 位置不参与，否则它们的 (0-V)² 会把
        # value 头往 0 拉，与"只在真实状态上拟合 V"不符。
        if self.ppo_cfg.value_clip:
            v_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.ppo_cfg.clip_param,
                self.ppo_cfg.clip_param,
            )
            v_se = torch.max(
                (new_values - returns) ** 2, (v_clipped - returns) ** 2
            )
        else:
            v_se = (new_values - returns) ** 2
        v_loss = 0.5 * (v_se * valid).sum() / valid_cnt

        # ---- 总损失 ----
        loss = (
            policy_loss
            + self.ppo_cfg.value_coef * v_loss
            - entropy_beta * total_entropy
        )

        # ---- explained variance（value 拟合质量，1.0 为完美）----
        # 同样按 valid_mask 计算（把 padded 位置置零），否则补齐的 0
        # 会虚高 returns 的方差、把该指标压低。
        returns_v = returns * valid
        values_v = new_values * valid
        var_y = returns_v.var(unbiased=False) + 1e-8
        var_res = (returns_v - values_v).var(unbiased=False)
        explained_var = 1.0 - var_res / var_y

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": v_loss.detach(),
            "entropy": total_entropy.detach(),
            "kl": kl.detach(),
            "clip_fraction": clip_fraction.detach(),
            "explained_variance": explained_var.detach(),
            "old_new_logp_gap": old_new_logp_gap.detach(),
            "entropy_beta": torch.as_tensor(entropy_beta, device=new_values.device),
            #: 本 batch 的有效决策占比（1.0 = 没有补齐）。用于确认
            #: valid_mask 真的透传到了 loss，而不是被静默丢弃。
            "valid_ratio": (valid_cnt / max(1, valid.numel())).detach(),
        }

    # -- 自检 -------------------------------------------------------------

    #: old/new log prob 每头平均绝对差的告警阈值。
    #: 纯浮点精度造成的差值应在 1e-4 量级；超过 0.05 基本可以断定
    #: 采样侧与训练侧跑的不是同一个前向函数。
    OLD_NEW_GAP_TOLERANCE = 0.05

    def _report_old_new_gap(self, losses: Dict[str, torch.Tensor], step: int) -> None:
        """记录并检查"参数未更新时 old/new log prob 是否一致"。

        PPO 的整个数学前提是 old_log_probs 与 new_log_probs 来自同一个策略
        函数。若采样侧（``env_runner.sample_action``：T=1、eval 模式）与
        训练侧（本模块：T=16、train 模式）跑的不是同一个函数，那么
        **参数一次都没更新时 ratio 就已经偏离 1**，KL 被凭空抬高并触发早停，
        clip / value_clip 失去基准 —— 表现为"loss 看着在动，胜率不动"。

        这里把该前提变成一条可观测指标（``train/old_new_logp_gap``），
        让这类问题在第一个 update 就暴露，而不是靠事后反推。
        """
        gap = float(losses["old_new_logp_gap"].item())
        # 存下来供 update() 放进 PPOStepResult —— 这样它也能进 metrics.jsonl，
        # 在看板上看到曲线（此前只进 TB，训练时只能到 stdout 里 grep）。
        self._last_old_new_gap = gap
        # 同时打一条 INFO 到控制台：这是判断"采样侧与训练侧是不是同一个前向"
        # 最直接的指标，训练时应当一眼可见（正常情况 ≈ 1e-4）。
        logger.info(
            "step=%d old_new_logp_gap=%.3e（参数未更新时应 ≈ 0；> %.2f 说明两侧前向不一致）",
            step, gap, self.OLD_NEW_GAP_TOLERANCE,
        )
        try:
            self.logger.log_metrics({"train/old_new_logp_gap": gap}, step)
        except Exception:      # noqa: BLE001 - 日志失败不应影响训练
            logger.debug("记录 old_new_logp_gap 失败", exc_info=True)
        if gap > self.OLD_NEW_GAP_TOLERANCE:
            logger.error(
                "step=%d 参数尚未更新，但 old/new log prob 每头平均绝对差已达 "
                "%.4f（阈值 %.3f）：采样侧与训练侧不是同一个前向。"
                "请检查 network.use_time_attention（应 False）、"
                "network.attn_dropout（应 0）、train.amp（建议 False）。"
                "此状态下 PPO 的 ratio / KL 早停 / value_clip 均不可信。",
                step, gap, self.OLD_NEW_GAP_TOLERANCE,
            )

    # -- 更新 -------------------------------------------------------------

    def update(self, batch: Dict[str, torch.Tensor], step: int,
               bc_alpha: float = 0.0,
               bc_expert_actions: Optional[torch.Tensor] = None) -> PPOStepResult:
        """对一个 batch 执行一次 PPO 更新（含 minibatch 切分与 KL 双阈值）。

        Args:
            batch: dict，含 features / legal_actions / actions / old_log_probs /
                   advantages / returns / old_values / lstm_state_init
            step:  全局更新步数（日志用）
            bc_alpha / bc_expert_actions: BC 辅助 loss（Phase 3 TODO，当前未实现）
        Returns:
            PPOStepResult
        """
        if bc_alpha > 0.0 and bc_expert_actions is not None:
            # 明确告知调用方该功能尚未实现，而不是静默忽略
            logger.warning("BC 辅助 loss 尚未实现（Phase 3），本次忽略 bc_alpha=%.3f", bc_alpha)

        self.network.train()

        # ---- 数据搬到 device ----
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        hw = batch.get("head_weights")
        if hw is None:
            if not self._warned_missing_head_weights:
                self._warned_missing_head_weights = True
                logger.warning(
                    "batch 中没有 head_weights（官方 weight0..5 = sub_action_mask[button]），"
                    "本次不做 head 屏蔽，等价于所有头都参与 loss —— 这不是官方做法。"
                    "env_runner.collect_trajectory 已经会返回 head_weights，"
                    "请检查训练脚本是否把它一并转成 tensor 传进 update()。"
                )
        elif not isinstance(hw, torch.Tensor):
            # 上游只做了"按固定 key 列表"的 numpy->tensor 转换时，
            # 新增的 head_weights 会以 ndarray 形式漏进来；这里兜底，
            # 避免它在下面被 CUDA 上的 index tensor 索引时报出莫名其妙的错。
            batch["head_weights"] = torch.as_tensor(
                np.asarray(hw), dtype=torch.float32
            ).to(self.device)

        if batch.get("head_weights") is not None and not self._logged_head_weights:
            self._logged_head_weights = True
            hw = batch["head_weights"]
            logger.info(
                "head 屏蔽已生效：head_weights %s，被屏蔽(权重为0)的头占比 %.1f%%"
                "（官方 weight0..5 = sub_action_mask[button]）",
                tuple(hw.shape),
                100.0 * float((hw == 0).float().mean().item()),
            )

        features = batch["features"]                     # (B, T, 725)
        B, T = features.shape[0], features.shape[1]

        # ---- 优势：batch 内 z-score（PPO 标准做法，policy 只关心相对优势）----
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # ---- 回报：RunningRewardNormalizer 在线归一化（替代逐 batch z-score）----
        # 原实现每 batch 各做一次 z-score，value 回归目标的尺度随 batch 漂移
        # （单局回报绝对值可达数百），V 网络永远在追一个移动靶，vloss 不稳、
        # 梯度爆炸（实测 grad_norm=inf）。改用跨 batch 的运行统计量后，
        # 回归目标是全局一致的零均值单位方差分布。
        if self.ppo_cfg.normalize_returns:
            returns_np = batch["returns"].detach().float().cpu().numpy()
            if np.isfinite(returns_np).all():
                # 先用本批数据在线更新统计量，再用更新后的统计量归一化
                self.return_normalizer.update(returns_np)
            else:
                # returns 含 inf/NaN（上游奖励或 value 异常）：跳过统计量更新，
                # 否则 Welford 的均值/方差被永久污染，后续所有 batch 全废。
                # 本批仍用旧统计量归一化（结果同样非有限），由下面的
                # 梯度有限性检查把受影响 minibatch 整体丢弃。
                logger.error(
                    "step=%d returns 含非有限值（%d/%d），跳过在线统计量更新",
                    step, int((~np.isfinite(returns_np)).sum()), returns_np.size,
                )
            batch["returns"] = torch.as_tensor(
                self.return_normalizer.normalize(returns_np),
                dtype=torch.float32,
                device=self.device,
            )
            # 关键：old_values 必须用同一组统计量做仿射变换。否则 value_clip
            # 在两个尺度之间失效：new_values≈O(1)，old_values≈O(100)，
            # v_clipped 与 returns 的平方误差巨大且对 new_values 无梯度，
            # V 网络学不动（vloss 高居不下但梯度传不回去）。
            batch["values"] = torch.as_tensor(
                self.return_normalizer.normalize(
                    batch["values"].detach().float().cpu().numpy()
                ),
                dtype=torch.float32,
                device=self.device,
            )

        # minibatch 以"帧"为单位申明，这里换算成序列条数：
        # 每条序列含 T 帧，故 mb_seqs = minibatch_size // T（至少 1 条）
        mb_seqs = max(1, min(self.ppo_cfg.minibatch_size // max(1, T), B))

        def _take(x, idx, dim=0):
            """按序列维取 minibatch；支持 tensor 与 (h, c) 元组。

            lstm_state_init 的 (h, c) 形状是 (1, B, H)，序列维在 dim 1
            （features 等是 (B, T, ...)，序列维在 dim 0），调用时需显式传 dim。
            """
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.index_select(dim, idx)
            if isinstance(x, (tuple, list)):
                return type(x)(_take(e, idx, dim) for e in x)
            return x

        early_stopped = False
        first_eval = True          # 本 update 的第一次前向（参数尚未更新）
        last = None
        grad_norm_value = 0.0
        num_minibatches = 0

        for epoch in range(self.ppo_cfg.ppo_epochs):
            # 每个 epoch 重新打乱序列顺序
            perm = torch.randperm(B, device=features.device)
            epoch_kl_exceeded_soft = False

            for start in range(0, B, mb_seqs):
                idx = perm[start:start + mb_seqs]

                mb = {
                    "features": features[idx],
                    "legal_actions": batch["legal_actions"][idx],
                    "actions": batch["actions"][idx],
                    "old_log_probs": batch["log_probs"][idx],
                    "advantages": batch["advantages"][idx],
                    "returns": batch["returns"][idx],
                    "old_values": batch["values"][idx],
                    "head_weights": (
                        batch["head_weights"][idx]
                        if batch.get("head_weights") is not None else None
                    ),
                    "valid_mask": (
                        batch["valid_mask"][idx]
                        if batch.get("valid_mask") is not None else None
                    ),
                }
                mb_state = _take(batch.get("lstm_state_init"), idx, dim=1)

                # ⚠️ 全部用关键字传参：compute_loss 的参数表还在演进，
                #    位置传参会在新增参数时静默错位（曾把 valid_mask 塞进
                #    lstm_state_init 的位置）。
                losses = self.compute_loss(
                    features=mb["features"],
                    legal_actions=mb["legal_actions"],
                    actions=mb["actions"],
                    old_log_probs=mb["old_log_probs"],
                    advantages=mb["advantages"],   # 已在 batch 级标准化
                    returns=mb["returns"],         # 已在 batch 级标准化
                    old_values=mb["old_values"],
                    head_weights=mb["head_weights"],
                    valid_mask=mb["valid_mask"],
                    lstm_state_init=mb_state,
                )
                loss = losses["loss"]

                # ---- 一致性自检：只在本 update 的第一次前向（参数未更新）时做 ----
                if first_eval:
                    first_eval = False
                    self._report_old_new_gap(losses, step)

                # ---- AMP 反向：scale -> backward -> unscale -> clip -> step ----
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                # 裁剪位置必须是 unscale 之后、scaler.step 之前：
                # 未 unscale 的梯度被放大过，clip 阈值会失准；
                # clip_grad_norm_ 返回的是**裁剪前**的总范数（诊断用）。
                # max_grad_norm=0.5 配合归一化后的 O(1) 损失：
                # 正常情况下 pre-clip 范数应在 0.1~2，超过 10 说明
                # 本 minibatch 有异常样本。
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.ppo_cfg.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    # 梯度非有限（fp16 溢出 / 输入被 NaN 污染）：跳过本次参数
                    # 更新，避免 NaN 权重悄悄毁掉整个训练。
                    # ⚠️ 即使跳过 step，也必须调 scaler.update()：
                    # unscale_() 会在 scaler 内部打"已 unscale"标记，不调
                    # update() 复位的话，下一个 minibatch 的 unscale_() 直接抛
                    # "unscale_() has already been called"（实测 step=18 崩溃）。
                    # update() 还会下调 scale，降低后续 inf 的概率。
                    bad_inputs = [
                        name for name, t in (
                            ("features", features),
                            ("advantages", batch["advantages"]),
                            ("returns", batch["returns"]),
                            ("values", batch["values"]),
                            ("log_probs", batch["log_probs"]),
                        )
                        if not torch.isfinite(t).all()
                    ]
                    logger.error(
                        "step=%d 梯度非有限（grad_norm=%s），跳过本 minibatch 更新。"
                        "非有限输入：%s",
                        step, float(grad_norm),
                        bad_inputs or "无（溢出发生在网络前向/反向内部）",
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    num_minibatches += 1
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()

                last = losses
                grad_norm_value = float(grad_norm.item())
                num_minibatches += 1

                # ---- KL 双阈值 ----
                kl_val = float(losses["kl"].item())
                if kl_val > self.ppo_cfg.kl_max:
                    # 硬阈值：策略已经跑偏，立即终止整个 update
                    logger.info(
                        "PPO KL 硬停止 @ step=%d epoch=%d kl=%.4f > kl_max=%.4f",
                        step, epoch, kl_val, self.ppo_cfg.kl_max,
                    )
                    early_stopped = True
                    break
                if kl_val > self.ppo_cfg.kl_target:
                    # 软阈值：本 epoch 剩下的 minibatch 不再更新
                    logger.debug(
                        "PPO KL 软停止 @ step=%d epoch=%d kl=%.4f > kl_target=%.4f",
                        step, epoch, kl_val, self.ppo_cfg.kl_target,
                    )
                    epoch_kl_exceeded_soft = True
                    break

            if early_stopped:
                break
            if epoch_kl_exceeded_soft:
                # 软阈值只跳过一个 epoch，继续下一个 epoch（此时策略已更新过，
                # 下一轮通常 KL 会回落）
                continue

        # ---- LR 调度 ----
        if self._lr_schedule is not None:
            self._lr_schedule.step()
        cur_lr = self.optimizer.param_groups[0]["lr"]

        # ---- VRAM 探测（内部可能抛 RuntimeError 表示 OOM）----
        vram_status = self.vram_guard.check(step=step)

        if last is None:
            # 所有 minibatch 的梯度都非有限：本步不更新，返回全零结果
            logger.error("step=%d 所有 minibatch 梯度均非有限，本步跳过更新", step)
            return PPOStepResult(
                policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0,
                grad_norm=0.0, lr=cur_lr, clip_fraction=0.0,
                explained_variance=0.0, entropy_beta=0.0,
                num_minibatches=num_minibatches, early_stopped=True,
            )

        result = PPOStepResult(
            policy_loss=float(last["policy_loss"].item()),
            value_loss=float(last["value_loss"].item()),
            entropy=float(last["entropy"].item()),
            kl=float(last["kl"].item()),
            grad_norm=grad_norm_value,
            lr=cur_lr,
            clip_fraction=float(last["clip_fraction"].item()),
            explained_variance=float(last["explained_variance"].item()),
            entropy_beta=float(last["entropy_beta"].item()),
            num_minibatches=num_minibatches,
            early_stopped=early_stopped,
            valid_ratio=float(last.get("valid_ratio", torch.tensor(1.0)).item()),
            old_new_logp_gap=float(self._last_old_new_gap),
        )

        self.logger.log_train_step(
            policy_loss=result.policy_loss,
            value_loss=result.value_loss,
            entropy=result.entropy,
            kl=result.kl,
            grad_norm=result.grad_norm,
            lr=result.lr,
            step=step,
            extra={
                "clip_fraction": result.clip_fraction,
                "explained_variance": result.explained_variance,
                "entropy_beta": result.entropy_beta,
                "num_minibatches": result.num_minibatches,
                "early_stopped": int(result.early_stopped),
                **vram_status.to_dict(),
            },
        )

        self._total_updates += 1
        return result

    # -- 存档 -------------------------------------------------------------

    def save(self, path: str):
        """保存 checkpoint（网络 / 优化器 / scaler / 步数）。"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "total_updates": self._total_updates,
        }, path)
        logger.info("已保存 checkpoint 到 %s", path)

    def load(self, path: str):
        """加载 checkpoint。"""
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self._total_updates = ckpt["total_updates"]
        logger.info("已从 %s 加载 checkpoint", path)
