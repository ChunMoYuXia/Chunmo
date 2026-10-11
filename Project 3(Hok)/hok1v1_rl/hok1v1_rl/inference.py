"""CPU 单步推理加速；训练仍使用原来的网络与梯度计算。

JIT 把固定形状 (1, 1, 725) 的前向过程保存成计算图，减少 Python 调用。
freeze 会固定权重，因此每次同步新策略后必须重新构建，不能沿用旧图。
"""
import torch
from torch import nn

from .action_mask import LABEL_SIZE_LIST


class _SingleStepNetwork(nn.Module):
    """trace 使用张量元组输出，避开网络中的 dict/list 混合结构。"""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, feature, hidden, cell):
        output = self.network(feature, (hidden, cell))
        next_hidden, next_cell = output["lstm_state_new"]
        return (torch.cat(output["logits_list"], dim=-1), output["value"],
                next_hidden, next_cell)


class CPUInference:
    """仅供采样使用的权重快照，不参与反向传播，也不写入 checkpoint。"""
    @torch.no_grad()
    def __init__(self, network):
        if next(network.parameters()).device.type != "cpu":
            raise ValueError("jit 推理目前仅支持 CPU 采样；GPU 请使用 eager")
        network.eval()
        feature = torch.zeros(1, 1, 725)
        hidden, cell = network.init_hidden(1, torch.device("cpu"))
        traced = torch.jit.trace(_SingleStepNetwork(network).eval(),
                                 (feature, hidden, cell), check_trace=False)
        self.graph = torch.jit.freeze(traced)
        # 首次图优化也在权重同步阶段完成，避免首个动作突然变慢。
        for _ in range(2):
            self.graph(feature, hidden, cell)

    @torch.no_grad()
    def __call__(self, feature, state):
        if feature.shape != (1, 1, 725):
            raise ValueError("jit 推理只接受单个环境的单步观测 (1,1,725)")
        logits, value, hidden, cell = self.graph(feature, *state)
        return {"logits_list": torch.split(logits, LABEL_SIZE_LIST, dim=-1),
                "value": value, "lstm_state_new": (hidden, cell)}
