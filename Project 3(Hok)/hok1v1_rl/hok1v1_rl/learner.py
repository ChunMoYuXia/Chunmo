"""PPO 更新：重放 LSTM 序列，按有效动作头计算损失与 KL。"""
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .action_mask import masked_softmax, probs_to_log_probs, split_legal_action_compressed
from .config import Config, ENVIRONMENT_CONTRACT
from .network import configure_float32

CHECKPOINT_VERSION = 2


def normalize_advantages(batch):
    """只用真实决策估计均值和方差，不让 padding 改变梯度尺度。"""
    batch = dict(batch)
    valid = batch["valid_mask"].bool()
    advantage = batch["advantages"][valid]
    normalized = (batch["advantages"] - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    batch["advantages"] = torch.where(valid, normalized, 0.0)
    return batch

@dataclass
class PPOStepResult:
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
    valid_ratio: float
    old_new_logp_gap: float
    policy_grad_norm: float = 0.0
    value_grad_norm: float = 0.0
    policy_head_grad_norm: float = 0.0
    shared_policy_grad_norm: float = 0.0
    shared_value_grad_norm: float = 0.0
    shared_grad_cosine: float = 0.0
    estimated_global_clip_scale: float = 1.0
    policy_head_clip_scale: float = 1.0
    old_new_value_gap: float = 0.0


def read_checkpoint(path, device="cpu"):
    # 仅加载自己生成、可信的本地 checkpoint；里面含 Python/NumPy RNG 状态。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("旧 checkpoint 的混合价值头与新结构不兼容。请保留旧模型，使用新 run 重新训练。")
    return checkpoint


class PPOLearner:
    """六头沿用逐头 clipped ratio；价值头直接回归 GAE return。"""
    def __init__(self, network, cfg, logger_obj=None, vram_guard=None):
        self.network, self.cfg = network, cfg
        self.device = next(network.parameters()).device
        configure_float32(self.device)
        self.logger, self.vram_guard = logger_obj, vram_guard
        self.optimizer = torch.optim.Adam(network.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self._total_updates = 0
        self._lr_total_steps = 1
        named = list(network.named_parameters())
        self.gradient_groups = [
            [p for name, p in named if name.startswith("action_head.")],
            [p for name, p in named if name.startswith("value_head.")],
            [p for name, p in named if not name.startswith(("action_head.", "value_head."))],
        ]

    def set_lr_schedule(self, total_steps):
        self._lr_total_steps = max(1, total_steps)

    def entropy_beta_at(self):
        progress = min(1, self._total_updates / self._lr_total_steps)
        p = self.cfg.ppo
        return p.entropy_beta_start + (p.entropy_beta_end - p.entropy_beta_start) * progress

    def compute_loss(self, features, legal_actions, actions, old_log_probs,
                     advantages, returns, old_values, head_weights=None,
                     valid_mask=None, lstm_state_init=None, entropy_beta=None):
        """padding 与无效动作头同时从损失、KL 和诊断的分母里去掉。"""
        out = self.network(features, lstm_state_init)
        valid = torch.ones_like(advantages) if valid_mask is None else valid_mask.float()
        heads = torch.ones_like(old_log_probs) if head_weights is None else head_weights.float()
        weights = heads * valid.unsqueeze(-1)
        # 补齐处无动作也不进入 loss，用任意非空 mask 避免误报 SDK 缺失动作。
        legal_actions = torch.where(valid.unsqueeze(-1) > 0, legal_actions, torch.ones_like(legal_actions))
        masks = split_legal_action_compressed(legal_actions)
        probs = [masked_softmax(logits, mask) for logits, mask in zip(out["logits_list"], masks)]
        logp = torch.stack([probs_to_log_probs(prob.gather(-1, actions[..., i:i+1]).squeeze(-1))
                            for i, prob in enumerate(probs)], dim=-1)
        # 先置零无效位置，避免极大 log ratio 在 exp 后形成 inf×0。
        difference = torch.where(weights > 0, logp - old_log_probs, torch.zeros_like(logp))
        ratio = difference.exp()
        p = self.cfg.ppo
        advantage = advantages.unsqueeze(-1)
        clipped = ratio.clamp(1 - p.clip_param, 1 + p.clip_param)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        valid_count, head_count = valid.sum().clamp_min(1), weights.sum().clamp_min(1)
        # 每次决策累加实际生效的动作头；不会让空动作头分走策略梯度。
        policy_loss = -(surrogate * weights).sum() / valid_count
        entropies = torch.stack([-(prob * probs_to_log_probs(prob)).sum(-1) for prob in probs], -1)
        entropy_loss = (entropies * weights).sum() / valid_count
        error = (out["value"] - returns).square()
        if p.value_clip:
            clipped_value = old_values + (out["value"] - old_values).clamp(-p.clip_param, p.clip_param)
            error = torch.maximum(error, (clipped_value - returns).square())
        value_loss = 0.5 * (error * valid).sum() / valid_count
        beta = self.entropy_beta_at() if entropy_beta is None else entropy_beta
        total_loss = policy_loss + p.value_coef * value_loss - beta * entropy_loss
        kl = ((ratio - 1 - difference) * weights).sum() / head_count
        gap = (difference.abs() * weights).sum() / head_count
        clip_fraction = (((ratio - 1).abs() > p.clip_param) * weights).sum() / head_count
        return {"loss": total_loss, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": (entropies * weights).sum() / head_count,
                "kl": kl, "old_new_logp_gap": gap,
                "old_new_logp_max": difference.abs().max(),
                "clip_fraction": clip_fraction, "new_values": out["value"],
                "valid_count": valid_count, "head_count": head_count}

    def _loss(self, batch, indices):
        keys = {"features": "features", "legal_actions": "legal_actions", "actions": "actions",
                "old_log_probs": "log_probs", "advantages": "advantages", "returns": "returns",
                "old_values": "values", "head_weights": "head_weights", "valid_mask": "valid_mask"}
        args = {name: batch[key][indices].to(self.device) for name, key in keys.items()}
        args["lstm_state_init"] = tuple(s[:, indices].to(self.device) for s in batch["lstm_state_init"])
        return self.compute_loss(**args)

    @torch.no_grad()
    def _diagnose(self, batch, chunks_per_batch):
        totals, denominators, predicted = {}, {}, []
        maximum_gap = 0.0
        for start in range(0, len(batch["features"]), chunks_per_batch):
            indices = torch.arange(start, min(start + chunks_per_batch, len(batch["features"])))
            losses = self._loss(batch, indices)
            maximum_gap = max(maximum_gap, float(losses["old_new_logp_max"]))
            for key in ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "old_new_logp_gap"):
                weight = float(losses["head_count"] if key in ("entropy", "kl", "clip_fraction", "old_new_logp_gap") else losses["valid_count"])
                totals[key] = totals.get(key, 0.0) + float(losses[key]) * weight
                denominators[key] = denominators.get(key, 0.0) + weight
            predicted.append(losses["new_values"].cpu())
        metrics = {key: value / denominators[key] for key, value in totals.items()}
        metrics["old_new_logp_max"] = maximum_gap
        values = torch.cat(predicted)
        valid = batch["valid_mask"].bool()
        value_difference = (values[valid] - batch["values"][valid]).abs()
        metrics["old_new_value_max"] = float(value_difference.max())
        # CPU/JIT/CUDA 重放允许少量数值误差；更新前才要求一致。
        metrics["old_new_value_scaled_max"] = float(
            (value_difference / (1e-3 + 1e-4 * values[valid].abs())).max())
        targets = batch["returns"][valid]
        variance = targets.var(unbiased=False)
        metrics["explained_variance"] = float(1 - (targets - values[valid]).var(unbiased=False) / variance) if variance > 1e-8 else 0.0
        return metrics

    def update(self, batch, step=None, total_steps=None):
        if total_steps:
            self.set_lr_schedule(total_steps)
        self.network.train()  # 网络无 dropout，train/eval 前向一致
        valid = batch["valid_mask"].bool()
        if not valid.any():
            raise ValueError("batch 没有有效决策")
        for name, tensor in batch.items():
            tensors = tensor if isinstance(tensor, tuple) else (tensor,)
            if any(not torch.isfinite(item).all() for item in tensors):
                raise FloatingPointError(f"batch.{name} 含 NaN/Inf")
        batch = normalize_advantages(batch)
        p = self.cfg.ppo
        progress = min(1, self._total_updates / self._lr_total_steps)
        lr = p.lr + (p.lr_end - p.lr) * progress
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.vram_guard:
            self.vram_guard.check(self._total_updates)
        chunk_size = max(1, p.minibatch_size // batch["features"].shape[1])
        initial = self._diagnose(batch, chunk_size)
        gap = initial["old_new_logp_max"]
        if not np.isfinite(gap) or gap > 1e-3:
            raise RuntimeError(f"更新前 old/new logp 差 {gap:.6f}，采样/训练不一致，已拒绝更新")
        if not np.isfinite(initial["old_new_value_scaled_max"]) or initial["old_new_value_scaled_max"] > 1:
            raise RuntimeError(f"更新前 old/new value 差 {initial['old_new_value_max']:.6f}，轨迹价值预测不一致，已拒绝更新")
        gradient_diagnostics = self.diagnose_gradients(batch, torch.arange(min(chunk_size, len(batch["features"]))))
        gradient_norms, policy_clip_scales, stopped = [], [], False
        for epoch in range(p.ppo_epochs):
            order = torch.randperm(len(batch["features"]))
            for indices in order.split(chunk_size):
                losses = self._loss(batch, indices)
                if not torch.isfinite(losses["loss"]) or not torch.isfinite(losses["kl"]):
                    raise FloatingPointError("PPO loss/KL 非有限值，停止训练")
                # 在这一步更新之前检查；触发后终止整个 update，不在下个 epoch 继续。
                if losses["kl"].item() > p.kl_target:
                    stopped = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad, policy_clip_scale = self._clip_gradients()
                self.optimizer.step()
                gradient_norms.append(float(grad))
                policy_clip_scales.append(policy_clip_scale)
            if stopped:
                break
        # 返回完整 batch 的更新后指标，不再只报告最后一个 minibatch。
        metrics = self._diagnose(batch, chunk_size)
        result = PPOStepResult(**{k: metrics[k] for k in ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "explained_variance")},
                               grad_norm=float(np.mean(gradient_norms)) if gradient_norms else 0.0,
                               lr=lr, entropy_beta=self.entropy_beta_at(), num_minibatches=len(gradient_norms),
                               early_stopped=stopped, valid_ratio=float(valid.float().mean()), old_new_logp_gap=gap,
                               old_new_value_gap=initial["old_new_value_max"],
                               policy_head_clip_scale=float(np.mean(policy_clip_scales)) if policy_clip_scales else 1.0,
                               **gradient_diagnostics)
        self._total_updates += 1
        if self.logger:
            self.logger.log_metrics({f"train/{k}": float(v) for k, v in asdict(result).items()}, self._total_updates)
        return result

    def _clip_gradients(self):
        """共享层仍接收策略与价值梯度；分组只隔离各组的裁剪尺度。"""
        limit = self.cfg.ppo.max_grad_norm
        if self.cfg.ppo.grad_clip_mode == "global":
            norm = nn.utils.clip_grad_norm_(self.network.parameters(), limit, error_if_nonfinite=True)
            return float(norm), min(1.0, limit / max(float(norm), 1e-12))
        norms = [nn.utils.clip_grad_norm_(group, limit, error_if_nonfinite=True)
                 for group in self.gradient_groups]
        # 三组参数不重叠，所以裁剪前的整网范数可以由平方和恢复。
        total = sum(float(norm) ** 2 for norm in norms) ** 0.5
        return total, min(1.0, limit / max(float(norms[0]), 1e-12))

    def diagnose_gradients(self, batch, indices, separate_entropy=False):
        """更新前第一批序列的梯度诊断，不写入 .grad，不执行 optimizer.step。

        policy 包含熵项，value 包含 value_coef，二者与实际优化目标一致。
        这是一个 minibatch 的诊断，不能代替整个训练过程的因果实验。
        """
        named = [(name, p) for name, p in self.network.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        was_training = self.network.training
        self.network.train()  # cuDNN LSTM 的 backward 要求训练模式；本网络无 dropout。
        try:
            with torch.enable_grad():
                losses = self._loss(batch, indices)
                policy_objective = losses["loss"] - self.cfg.ppo.value_coef * losses["value_loss"]
                policy = torch.autograd.grad(policy_objective, parameters, retain_graph=True, allow_unused=True)
                surrogate = (torch.autograd.grad(losses["policy_loss"], parameters,
                             retain_graph=True, allow_unused=True) if separate_entropy else None)
                value = torch.autograd.grad(self.cfg.ppo.value_coef * losses["value_loss"], parameters, allow_unused=True)
        finally:
            self.network.train(was_training)
        def square_sum(gradients, predicate=lambda name: True):
            return sum(float(g.detach().square().sum()) for (name, _), g in zip(named, gradients)
                       if g is not None and predicate(name))
        shared = lambda name: name.startswith(("encoder.", "lstm."))
        policy_square, value_square = square_sum(policy, shared), square_sum(value, shared)
        dot, combined_square = 0.0, 0.0
        for (name, _), actor, critic in zip(named, policy, value):
            if actor is not None and critic is not None:
                combined_square += float((actor.detach() + critic.detach()).square().sum())
                if shared(name):
                    dot += float((actor.detach() * critic.detach()).sum())
            elif actor is not None or critic is not None:
                gradient = actor if actor is not None else critic
                combined_square += float(gradient.detach().square().sum())
        result = {"policy_grad_norm": square_sum(policy) ** 0.5,
                "value_grad_norm": square_sum(value) ** 0.5,
                "policy_head_grad_norm": square_sum(policy, lambda name: name.startswith("action_head.")) ** 0.5,
                "shared_policy_grad_norm": policy_square ** 0.5,
                "shared_value_grad_norm": value_square ** 0.5,
                "shared_grad_cosine": dot / max((policy_square * value_square) ** 0.5, 1e-12),
                "estimated_global_clip_scale": min(1.0, self.cfg.ppo.max_grad_norm / max(combined_square ** 0.5, 1e-12))}
        if surrogate is not None:
            entropy = tuple((p if s is None else -s if p is None else p - s)
                            for p, s in zip(policy, surrogate))
            result.update(surrogate_grad_norm=square_sum(surrogate) ** .5,
                          entropy_grad_norm=square_sum(entropy) ** .5,
                          shared_surrogate_grad_norm=square_sum(surrogate, shared) ** .5,
                          shared_entropy_grad_norm=square_sum(entropy, shared) ** .5)
        return result

    def save(self, path, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"version": CHECKPOINT_VERSION, "environment_contract": ENVIRONMENT_CONTRACT,
                 "config": self.cfg.to_dict(),
                 "network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "updates": self._total_updates, "schedule_total": self._lr_total_steps,
                 "training_state": training_state or {},
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def _check_checkpoint_config(self, state, allow_ppo_changes=False, allow_reward_changes=False):
        if not allow_ppo_changes and state.get("environment_contract") != ENVIRONMENT_CONTRACT:
            raise ValueError("checkpoint 采样契约不同或缺失（当前包含价值预测记录修复）；请用 --init-from 开启新实验，不可严格 --resume")
        # 新增配置字段时，用默认值补齐旧 checkpoint，避免无关字段阻止正常续训。
        previous = Config.from_dict(state["config"]).to_dict()
        sections = ("network",) if allow_ppo_changes else ("network", "ppo")
        if not allow_reward_changes:
            sections += ("reward",)
        if not allow_ppo_changes:
            sections += ("action",)
        for section in sections:
            if previous[section] != self.cfg.to_dict()[section]:
                raise ValueError(f"续训 {section} 配置与 checkpoint 不同；请使用 checkpoint 配置")
        task_fields = ("hero_ids", "opponent_hero_ids", "max_frame_num",
                       "predict_frequency", "reward_time_coef", "max_decisions")
        for name in task_fields:
            if previous["env"][name] != getattr(self.cfg.env, name):
                raise ValueError(f"续训 env.{name} 与 checkpoint 不同，不能混合训练任务")

    def initialize_from(self, path, allow_reward_changes=False):
        """权重初始化；奖励迁移须显式启用，Adam、计数和随机流重置。"""
        if self._total_updates or self.optimizer.state:
            raise ValueError("权重初始化只能用于尚未更新的新 learner")
        state = read_checkpoint(path)
        self._check_checkpoint_config(state, allow_ppo_changes=True, allow_reward_changes=allow_reward_changes)
        self.network.load_state_dict(state["network"])
        return {"mode": "weights_only", "source_checkpoint": str(Path(path).resolve()),
                "source_updates": state["updates"],
                "source_action": Config.from_dict(state["config"]).to_dict()["action"],
                "source_reward": Config.from_dict(state["config"]).to_dict()["reward"],
                "reward": self.cfg.to_dict()["reward"],
                "reward_changed": Config.from_dict(state["config"]).to_dict()["reward"] != self.cfg.to_dict()["reward"],
                "action": self.cfg.to_dict()["action"]}

    def load(self, path):
        state = read_checkpoint(path)
        self._check_checkpoint_config(state)
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._total_updates = state["updates"]
        self._lr_total_steps = state["schedule_total"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if self.device.type == "cuda" and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        return state["training_state"]
