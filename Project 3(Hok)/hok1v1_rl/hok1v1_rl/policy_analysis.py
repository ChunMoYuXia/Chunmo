"""分析真实 batch 的动作信用与各头探索；这些条件统计不等于动作因果收益。"""
import torch
from .action_mask import masked_softmax, probs_to_log_probs, split_legal_action_compressed


def action_advantages(batch):
    """只统计有效步；由 return-value 重建原始 GAE，兼容已标准化的诊断 batch。

    不同动作可能出现在不同状态，均值差不能证明替换动作后的收益。
    额外按英雄拆分，避免把阵容难度直接当成动作偏好。
    """
    valid = batch["valid_mask"].bool()
    raw = (batch["returns"] - batch["values"])[valid].float()
    if not raw.numel():
        raise ValueError("没有有效决策可供动作优势诊断")
    normalized = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-8)
    buttons = batch["actions"][..., 0][valid]
    values, returns = batch["values"][valid], batch["returns"][valid]
    head_counts = batch["head_weights"][valid].sum(-1)
    legal = batch["legal_actions"][valid]
    idle_alternative = (buttons == 1) & (legal[:, 2:12] > 0).any(-1)

    def summarize(selection):
        count = int(selection.sum())
        if not count:
            return {"decisions": 0}
        return {"decisions": count, "raw_advantage_mean": float(raw[selection].mean()),
                "normalized_advantage_mean": float(normalized[selection].mean()),
                "positive_advantage_fraction": float((normalized[selection] > 0).float().mean()),
                "return_mean": float(returns[selection].mean()),
                "value_mean": float(values[selection].mean()),
                "active_heads_mean": float(head_counts[selection].mean())}

    by_button = {str(i): summarize(buttons == i) for i in range(12)}
    by_hero = {}
    features = batch["features"][valid]
    if features.shape[-1] == 725:
        from .config import HERO_IDS
        identity = features[:, 705:]
        known = (identity.sum(-1) - 1).abs() < 1e-5
        for index, hero in enumerate(HERO_IDS):
            selected = known & (identity[:, index] > .5)
            if selected.any():
                by_hero[str(hero)] = {str(i): summarize(selected & (buttons == i))
                                      for i in range(12)}
    return {"valid_decisions": raw.numel(), "padding_steps": int((~valid).sum()),
            "raw_advantage_mean": float(raw.mean()),
            "raw_advantage_std": float(raw.std(unbiased=False)),
            "by_button": by_button, "idle_with_alternative": summarize(idle_alternative),
            "by_hero": by_hero}


@torch.no_grad()
def head_exploration(learner, batch):
    """逐头按实际生效次数加权；不把 padding 或未生效的动作头算作探索。

    按合法选项数归一化的熵仅统计至少有两个合法选项的决策。
    """
    entropy_sum = torch.zeros(6, dtype=torch.float64)
    active_count = torch.zeros(6, dtype=torch.float64)
    normalized_sum = torch.zeros(6, dtype=torch.float64)
    exploratory_count = torch.zeros(6, dtype=torch.float64)
    top1_sum = torch.zeros(6, dtype=torch.float64)
    size = max(1, learner.cfg.ppo.minibatch_size // batch["features"].shape[1])
    was_training = learner.network.training
    learner.network.eval()
    try:
        for start in range(0, len(batch["features"]), size):
            sl = slice(start, start + size)
            features = batch["features"][sl].to(learner.device)
            hidden = tuple(s[:, sl].to(learner.device) for s in batch["lstm_state_init"])
            output = learner.network(features, hidden)
            valid = batch["valid_mask"][sl].to(learner.device).float()
            heads = batch["head_weights"][sl].to(learner.device).float() * valid.unsqueeze(-1)
            legal = batch["legal_actions"][sl].to(learner.device)
            legal = torch.where(valid.unsqueeze(-1) > 0, legal, torch.ones_like(legal))
            for head, mask in enumerate(split_legal_action_compressed(legal)):
                probability = masked_softmax(output["logits_list"][head], mask)
                entropy = -(probability * probs_to_log_probs(probability)).sum(-1)
                weight = heads[..., head]
                options = mask.sum(-1)
                exploratory = weight * (options > 1)
                entropy_sum[head] += float((entropy * weight).sum())
                active_count[head] += float(weight.sum())
                top1_sum[head] += float((probability.max(-1).values * weight).sum())
                normalized_sum[head] += float((entropy / options.clamp_min(2).log() * exploratory).sum())
                exploratory_count[head] += float(exploratory.sum())
    finally:
        learner.network.train(was_training)
    return {str(i): {"active_decisions": int(active_count[i]),
                     "entropy_mean": float(entropy_sum[i] / active_count[i]) if active_count[i] else None,
                     "top1_probability_mean": float(top1_sum[i] / active_count[i]) if active_count[i] else None,
                     "normalized_entropy_mean": float(normalized_sum[i] / exploratory_count[i]) if exploratory_count[i] else None}
            for i in range(6)}
