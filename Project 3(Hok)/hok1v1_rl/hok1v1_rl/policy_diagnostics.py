"""记录合法动作、概率和选择比例。概率很接近时，argmax 也可能长期选同一项。"""
import numpy as np


class PolicyDiagnostics:
    def __init__(self):
        self.decisions = 0
        self.legal = np.zeros(12, dtype=np.int64)
        self.probability = np.zeros(12, dtype=np.float64)
        self.selected = np.zeros(12, dtype=np.int64)
        self.greedy = np.zeros(12, dtype=np.int64)
        self.idle_with_alternative = 0
        self.near_ties = 0
        self.margin = 0.0

    def add(self, legal, probabilities, button):
        legal = np.asarray(legal[:12], dtype=bool)
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(12)
        self.decisions += 1
        self.legal += legal
        self.probability += probabilities
        self.selected[button] += 1
        self.greedy[int(probabilities.argmax())] += 1
        self.idle_with_alternative += button == 1 and legal[2:].any()
        top = np.sort(probabilities[legal])
        if len(top) >= 2:
            margin = float(top[-1] - top[-2])
            self.margin += margin
            self.near_ties += margin < 0.02

    def to_dict(self):
        n = max(1, self.decisions)
        return {"decisions": self.decisions, "legal_counts": self.legal.tolist(),
                "probability_sums": self.probability.tolist(),
                "selected_counts": self.selected.tolist(), "greedy_counts": self.greedy.tolist(),
                "legal_rate": (self.legal / n).tolist(),
                "probability_mean": (self.probability / n).tolist(),
                "probability_when_legal": [float(p / c) if c else None for p, c in zip(self.probability, self.legal)],
                "selection_rate": (self.selected / n).tolist(),
                "greedy_rate": (self.greedy / n).tolist(),
                "near_tie_rate": self.near_ties / n,
                "mean_top2_margin": self.margin / n,
                "idle_with_alternative_rate": float(self.idle_with_alternative / n)}


def merge_policy_diagnostics(games):
    """不同局长按决策数加权；旧日志没有诊断字段时保持为空。"""
    rows = [g["policy_diagnostics"] for g in games if g.get("policy_diagnostics")]
    if not rows:
        return {}
    total = PolicyDiagnostics()
    total.decisions = sum(r["decisions"] for r in rows)
    for attribute, key in (("legal", "legal_counts"), ("probability", "probability_sums"),
                           ("selected", "selected_counts"), ("greedy", "greedy_counts")):
        setattr(total, attribute, np.sum([r[key] for r in rows], axis=0))
    total.near_ties = sum(r["near_tie_rate"] * r["decisions"] for r in rows)
    total.margin = sum(r["mean_top2_margin"] * r["decisions"] for r in rows)
    total.idle_with_alternative = sum(r["idle_with_alternative_rate"] * r["decisions"] for r in rows)
    return total.to_dict()
