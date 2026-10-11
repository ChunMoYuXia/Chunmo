"""观测分块编码 → LSTM 记忆 → 六头动作 + 单个价值预测。

保持官方特征布局与目标槽位顺序，避免把不同单位的特征混在一起。
B 表示序列批次数，T 表示每条序列长度，H 表示 LSTM 隐层宽度。
"""
import math
from typing import List
import torch
from torch import nn
from .action_mask import LABEL_SIZE_LIST


def configure_float32(device):
    """CPU 采样和 CUDA 序列重放统一使用 FP32，避免 cuDNN TF32 价值误差。"""
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def make_fc(input_size, output_size, bias=True):
    layer = nn.Linear(input_size, output_size, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def make_mlp(sizes: List[int], non_linearity_last=False):
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(make_fc(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2 or non_linearity_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FeatureEncoder(nn.Module):
    """友敌单位共享编码主干，分别投影；小兵/建筑在单位维做 max-pool。"""
    def __init__(self, feature_dim=725, hidden_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.hero_main_mlp = make_mlp([14, 64, 32, 16])
        self.hero_mlp = make_mlp([235, 512, 256], non_linearity_last=True)
        self.hero_frd_fc = make_fc(256, 128)
        self.hero_emy_fc = make_fc(256, 128)
        self.soldier_mlp = make_mlp([18, 64, 64], non_linearity_last=True)
        self.soldier_frd_fc = make_fc(64, 32)
        self.soldier_emy_fc = make_fc(64, 32)
        self.organ_mlp = make_mlp([18, 64, 64], non_linearity_last=True)
        self.organ_frd_fc = make_fc(64, 32)
        self.organ_emy_fc = make_fc(64, 32)
        # 32×4 + 16 + 128×2 + 25 = 425。
        self.head_mlp = make_mlp([425, hidden_dim], non_linearity_last=True)

    def split_feature(self, feature):
        if not torch.jit.is_tracing() and feature.shape[-1] != self.feature_dim:
            raise ValueError(f"观测宽度应为 {self.feature_dim}，实际 {feature.shape[-1]}")
        leading = feature.shape[:-1]
        return {
            "hero_frd": feature[..., 0:235],
            "hero_emy": feature[..., 235:470],
            "hero_main": feature[..., 470:484],
            "soldier_frd": feature[..., 484:556].reshape(*leading, 4, 18),
            "soldier_emy": feature[..., 556:628].reshape(*leading, 4, 18),
            "organ_frd": feature[..., 628:664].reshape(*leading, 2, 18),
            "organ_emy": feature[..., 664:700].reshape(*leading, 2, 18),
            "global": feature[..., 700:725],  # 5 游戏全局特征 + 20 英雄身份
        }

    @staticmethod
    def _hero_branch(trunk, projection, feature):
        state = projection(trunk(feature))
        return state, state[..., -32:]  # 最后 32 维同时用于目标打分

    @staticmethod
    def _unit_branch(trunk, projection, features):
        units = projection(trunk(features))
        return units.max(dim=-2).values, units

    def forward(self, feature):
        blocks = self.split_feature(feature)
        main = self.hero_main_mlp(blocks["hero_main"])
        friend, friend_target = self._hero_branch(
            self.hero_mlp, self.hero_frd_fc, blocks["hero_frd"])
        enemy, enemy_target = self._hero_branch(
            self.hero_mlp, self.hero_emy_fc, blocks["hero_emy"])
        friend_soldiers, _ = self._unit_branch(
            self.soldier_mlp, self.soldier_frd_fc, blocks["soldier_frd"])
        enemy_soldiers, enemy_soldier_targets = self._unit_branch(
            self.soldier_mlp, self.soldier_emy_fc, blocks["soldier_emy"])
        friend_organs, _ = self._unit_branch(
            self.organ_mlp, self.organ_frd_fc, blocks["organ_frd"])
        enemy_organs, _ = self._unit_branch(
            self.organ_mlp, self.organ_emy_fc, blocks["organ_emy"])
        combined = torch.cat([
            friend_soldiers, enemy_soldiers, friend_organs, enemy_organs,
            main, friend, enemy, blocks["global"],
        ], dim=-1)
        empty_target = torch.full_like(enemy_target.unsqueeze(-2), 0.1)
        # 顺序由 gamecore 动作协议定义，不能任意重排。
        targets = torch.cat([
            empty_target, enemy_target.unsqueeze(-2), friend_target.unsqueeze(-2),
            enemy_soldier_targets, enemy_organs.unsqueeze(-2),
        ], dim=-2)
        return self.head_mlp(combined), targets


class ActionHead(nn.Module):
    """前五头预测动作标签；target 头用单位特征与 LSTM query 的点积打分。"""
    def __init__(self, hidden_dim=512, target_embed_dim=32):
        super().__init__()
        self.label_mlps = nn.ModuleList([
            make_mlp([hidden_dim, 64, size]) for size in LABEL_SIZE_LIST[:5]
        ])
        self.tar_query = make_fc(hidden_dim, target_embed_dim)
        self.target_embed_mlp = make_fc(target_embed_dim, target_embed_dim, bias=False)

    def forward(self, hidden, targets):
        logits = [head(hidden) for head in self.label_mlps]
        query = self.tar_query(hidden).unsqueeze(-1)
        units = self.target_embed_mlp(targets)
        logits.append(torch.matmul(units, query).squeeze(-1))
        return logits


class HoK1v1Network(nn.Module):
    """输入 (B,T,725)，输出六头 logits、V(s) 与新的 LSTM 状态。"""
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.lstm_hidden = cfg.network.lstm_hidden
        self.encoder = FeatureEncoder(725, self.lstm_hidden)
        self.lstm = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True)
        self.action_head = ActionHead(self.lstm_hidden, 32)
        self.value_head = make_mlp([self.lstm_hidden, 64, 1])
        # 初始策略接近均匀分布；critic 直接监督总回报。
        for head in self.action_head.label_mlps:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.action_head.tar_query.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1)

    def forward(self, feature, lstm_state=None):
        embedding, targets = self.encoder(feature)
        if lstm_state is None:
            lstm_state = self.init_hidden(feature.shape[0], feature.device)
        hidden, new_state = self.lstm(embedding, lstm_state)
        return {
            "logits_list": self.action_head(hidden, targets),
            "value": self.value_head(hidden).squeeze(-1),
            "lstm_state_new": new_state,
        }

    def init_hidden(self, batch_size, device):
        shape = (1, batch_size, self.lstm_hidden)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
