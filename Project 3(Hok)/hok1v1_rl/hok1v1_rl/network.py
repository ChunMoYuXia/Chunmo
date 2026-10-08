"""网络定义 —— LSTM + 4-head Self-Attention + Multi-head Value。

────────────────────────────────────────────────────────────────────────────
与官方 baseline 的关系（aiarena/1v1/common/algorithm_torch.py）
────────────────────────────────────────────────────────────────────────────
忠实复刻的部分：
  * 特征切分顺序（**这是原实现最严重的一处静默 bug，见下**）
  * 各分支 MLP 维度、``concat_dim = 425``
  * LSTM(input=512, hidden=512, num_layers=1, batch_first=True)
  * target 头用"单位 embedding × LSTM query 点积"产生 logits
  * 6 头动作空间 LABEL_SIZE_LIST = [12,16,16,16,16,8]

本项目额外增强（官方没有）：
  * LSTM 之后的 4-head self-attention
  * 6 头 Multi-head Value（官方是单个 value 头）

────────────────────────────────────────────────────────────────────────────
⚠️ 特征布局（725 维）——**必须按官方顺序**
────────────────────────────────────────────────────────────────────────────
官方切分代码（algorithm_torch.py:199-241，algorithm_tf.py:405-458 独立互证）::

    feature_vec_split_list = feature_vec.split(
        [all_hero_feature_dim,      # 484 = 235+235+14
         all_soldier_feature_dim,   # 144 = 72+72
         all_organ_feature_dim,     #  72 = 36+36
         global_feature_dim],       #  25
        dim=1)
    hero_vec_list = feature_vec_split_list[0].split([235, 235, 14], dim=1)
    _hero_frd, _hero_emy, _hero_main = hero_vec_list[0], hero_vec_list[1], hero_vec_list[2]

因此真实布局是::

    [  0 : 235] hero_frd      友方英雄 (235)
    [235 : 470] hero_emy      敌方英雄 (235)
    [470 : 484] hero_main     公共英雄信息 (14)
    [484 : 556] soldier_frd   友方小兵 (4 × 18)
    [556 : 628] soldier_emy   敌方小兵 (4 × 18)
    [628 : 664] organ_frd     友方建筑 (2 × 18)
    [664 : 700] organ_emy     敌方建筑 (2 × 18)
    [700 : 705] global        game_time 全局信息 (**5**)
    [705 : 725] hero_identity 我方英雄身份 one-hot (**20**，由 env_runner 追加)

⚠️ 关于 705 与 725（**已对照官方源码查清，不是 gamecore 版本差异**）
    本项目 gamecore 给出的原始观测确实是 **705** 维，
    705 - 484(英雄) - 144(小兵) - 72(建筑) = 5，即 global 只有 5 维 game_time。

    但官方 baseline 喂给网络的从来不是 705，而是 **725**：
    ``aiarena/1v1/actor/custom.py::Agent.append_hero_identity`` 会把
    我方英雄 id 转成 **20 维 one-hot 拼到观测末尾**::

        HERO_ID_INDEX_DICT = {112:0, 121:1, ..., 513:19}   # 恰好 20 个英雄
        state_dict["observation"] = np.concatenate((obs, hero_id_vec), axis=0)

    于是网络里的 ``DIM_OF_GLOBAL_INFO = [25] = 5(game_time) + 20(hero identity)``，
    观测总维度 725，``concat_dim`` 425（而非 705 时的 405）。

    ⚠️ 这不是可有可无的一维噪声：本项目每局都通过
    ``camp_iterator_1v1_roundrobin_camp_heroes`` **轮换英雄**，
    缺了这 20 维，同一个策略网络必须在"不知道自己是谁"的前提下
    学会 20 个英雄的打法 —— 这是训练效果差的一个独立原因。
    因此 ``env_runner.build_observation`` 会把这一 20 维补上，
    网络侧 ``GLOBAL_DIM`` 相应为 25。

    英雄块内部的顺序仍然沿用官方（frd | emy | main）：
    小兵/建筑段呈严格 18 维周期、起点正好 484，说明英雄块合计 484，
    与官方一致；段内顺序按官方源码。

**原实现错误地按 hero_main | hero_frd | hero_emy 切分前 484 维**：
因为 14 + 235 + 235 仍然等于 484，**不会报任何形状错误**，
但会把 hero_frd 的头部当 hero_main、把两个英雄的特征交叉混入错误分支，
且 hero_main 从未被单独提取 —— 训练会静默劣化且极难定位。

    注意 ``Config.SERI_VEC_SPLIT_SHAPE`` 只有 ``[(725,), (84,)]``，
    它**不携带段内顺序**，所以"按 SERI_VEC_SPLIT 切分"这句话本身不足以确定顺序，
    必须以上面的 ``feature_vec.split`` 代码为准。

────────────────────────────────────────────────────────────────────────────
⚠️ target 头的 8 个槽位语义（官方 algorithm_torch.py:244-332）
────────────────────────────────────────────────────────────────────────────
官方把 ``tar_embed_list`` 按固定顺序拼装（最后在 index 0 插入一个常量占位）::

    idx 0: 常量 0.1 * ones_like(...)          —— "无目标"占位
    idx 1: hero_emy_fc_out 的后 32 维          —— 敌方英雄
    idx 2: hero_frd_fc_out 的后 32 维          —— 友方英雄
    idx 3-6: soldier_emy_fc_out（4 个小兵各 32 维）
    idx 7: reshape_pool_emy_organ（敌方建筑池化后 32 维）

总计 8 个槽位，恰好等于 ``LABEL_SIZE_LIST[-1] = 8``，
与 legal_action 中 (12 button × 8 target) 的排布一一对应。

target logits 由下式得到::

    ulti_tar_embedding = target_embed_mlp(tar_embedding)          # (…, 8, 32)
    query              = lstm_tar_embed_mlp(lstm_out)             # (…, 32)
    logits             = matmul(ulti_tar_embedding, query[..., None]).squeeze(-1)
                                                                  # (…, 8)

即 **8 个槽位各自有一份从"该单位的观测特征"学出来的 embedding**，
再与"当前局面"的 query 做点积评分。原实现返回的是与单位无关的
``target_embed_mlp(hidden)``，属于占位逻辑。

AMP 友好：避免 in-place 操作；masked softmax 由 action_mask 模块负责。
"""

from typing import Dict, List, Optional, Tuple
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_mask import LABEL_SIZE_LIST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

def make_fc(inp_dim: int, out_dim: int, bias: bool = True) -> nn.Linear:
    """正交初始化的全连接层（与官方 ``make_fc_layer`` 对齐）。

    使用 orthogonal init + gain=sqrt(2)：对 ReLU 网络是经验最优，
    可防止训练初期梯度消失/爆炸。

    注意返回类型是 ``nn.Linear``；原实现把返回标注写成 ``nn.Sequential``，
    标注有误但不影响运行。
    """
    layer = nn.Linear(inp_dim, out_dim, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
    if bias:
        nn.init.constant_(layer.bias, 0.0)
    return layer


def make_mlp(dims: List[int], non_linearity_last: bool = False) -> nn.Sequential:
    """带 ReLU 与正交初始化的多层 MLP。

    Args:
        dims: [in, hidden1, ..., out]
        non_linearity_last: True 时最后一层后也加 ReLU（官方
            ``MLP(..., non_linearity_last=True)`` 的语义）
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(make_fc(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or non_linearity_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 分支维度常量（全部来自官方 DimConfig / algorithm_torch.py）
# ---------------------------------------------------------------------------

HERO_MAIN_DIM = 14
HERO_FRD_DIM = 235
HERO_EMY_DIM = 235
SOLDIER_DIM = 18
SOLDIER_NUM = 4          # DIM_OF_SOLDIER_1_10 / 11_20 各有 4 个小兵
ORGAN_DIM = 18
ORGAN_NUM = 2            # DIM_OF_ORGAN_1_2 / 3_4 各有 2 个建筑
GLOBAL_TIME_DIM = 5      # 官方 DIM_OF_GLOBAL_INFO 里的 5 维 game_time
#: 官方 aiarena/1v1/actor/custom.py 在观测末尾追加的 20 维**英雄身份 one-hot**。
#: 加上它才是官方 baseline 真正喂给网络的 25 维 global 段（5 + 20 = 25），
#: 观测总维度也才是 725。缺了它就只有 705（gamecore 原始观测）。
HERO_IDENTITY_DIM = 20
#: 官方 DIM_OF_GLOBAL_INFO = 25（合计 725）。
#: ⚠️ 本项目最初按 705（global=5）建网，结果丢掉了英雄身份。
#:    network 侧的 global 段必须与观测末尾的实际拼法一致：
#:    [700:705] 是 game_time，[705:725] 是 20 维英雄 one-hot。
GLOBAL_DIM = GLOBAL_TIME_DIM + HERO_IDENTITY_DIM   # 25

#: 由各分段推算出的观测总维度（应当等于 cfg.network.feature_dim）。
#: 484(英雄) + 144(小兵) + 72(建筑) + 25(global: 5 game_time + 20 英雄身份) = 725
FEATURE_DIM_EXPECTED = (
    HERO_FRD_DIM + HERO_EMY_DIM + HERO_MAIN_DIM
    + SOLDIER_DIM * SOLDIER_NUM * 2
    + ORGAN_DIM * ORGAN_NUM * 2
    + GLOBAL_DIM
)

#: 英雄分支 fc 输出 128 维，完整 128 维进状态表示 concat；
#: 只有 target embedding 取 split([96, 32]) 的后 32 维
#: （官方 algorithm_torch.py:255-258:
#:   ``_, split_1 = hero_emy_fc_out.split([96, 32], dim=1)`` 后
#:   ``hero_emy_result_list.append(hero_emy_fc_out)``，即完整 128 维进 concat）
HERO_FC_OUT = 128
HERO_STATE_PART = 96
HERO_TARGET_PART = 32

#: concat 后总维度 = 32×4(soldier/organ 各 2) + 16(hero_main) + 128×2(hero) + global
#: global=25（5 game_time + 20 英雄身份）时 = 128 + 16 + 256 + 25 = **425**，
#: 与官方 concat_dim 完全一致（旧代码按 global=5 得到 405，对不上官方）。
CONCAT_DIM = (
    32 * 2          # soldier_frd / soldier_emy 池化后各 32
    + 32 * 2        # organ_frd   / organ_emy   池化后各 32
    + 16            # hero_main
    + HERO_FC_OUT * 2   # hero_frd / hero_emy（各 128）
    + GLOBAL_DIM    # global（5 + 20 = 25）
)


# ---------------------------------------------------------------------------
# 特征编码器
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """把 725 维原始特征编码为 512 维 embedding，同时输出 8 个 target 槽位的 embedding。

    分支结构（与官方 algorithm_torch.py:67-158 对齐）::

        hero_main : 14  -> 64 -> 32 -> 16
        hero      : 235 -> 512 -> 256 -> 128   (友/敌共享 hero_mlp，各自独立 fc)
        soldier   : 18  -> 64  -> 64  -> 32    (友/敌共享 soldier_mlp，各自独立 fc)
        organ     : 18  -> 64  -> 64  -> 32    (友/敌共享 organ_mlp，各自独立 fc)
        global    : 25  (直通)
        concat    : 425 -> 512

    池化说明：小兵/建筑按"单位维"做 max-pool，因为单位数量可变且
    max-pool 对关键单位（如正在攻击我的那个小兵）的信号更敏感。
    英雄分支只有 1 个单位，池化是恒等操作。
    """

    def __init__(self, feature_dim: int = 725):
        super().__init__()
        self._feature_dim = feature_dim

        # hero_main：只有 1 个单位，直接 MLP
        self.hero_main_mlp = make_mlp([HERO_MAIN_DIM, 64, 32, 16])

        # hero：共享 trunk + 各自 fc（官方 hero_mlp / hero_frd_fc / hero_emy_fc）
        self.hero_mlp = make_mlp([HERO_FRD_DIM, 512, 256], non_linearity_last=True)
        self.hero_frd_fc = make_fc(256, HERO_FC_OUT)
        self.hero_emy_fc = make_fc(256, HERO_FC_OUT)

        # soldier：共享 trunk + 各自 fc
        self.soldier_mlp = make_mlp([SOLDIER_DIM, 64, 64], non_linearity_last=True)
        self.soldier_frd_fc = make_fc(64, 32)
        self.soldier_emy_fc = make_fc(64, 32)

        # organ：共享 trunk + 各自 fc
        self.organ_mlp = make_mlp([ORGAN_DIM, 64, 64], non_linearity_last=True)
        self.organ_frd_fc = make_fc(64, 32)
        self.organ_emy_fc = make_fc(64, 32)

        # public concat
        self.head_mlp = make_mlp([CONCAT_DIM, 512], non_linearity_last=True)

    # -- 特征切分 ---------------------------------------------------------

    def split_feature(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """按**官方顺序**把 (..., 725) 切分为具名分支。

        ⚠️ 顺序绝对不能改，原因见模块 docstring 中的布局表。

        小兵/建筑会 reshape 成 (..., n_units, unit_dim)，
        以便随后在单位维上做 max-pool。

        Returns:
            dict，键：hero_frd / hero_emy / hero_main /
                      soldier_frd / soldier_emy / organ_frd / organ_emy / global
        """
        assert feature.shape[-1] == self._feature_dim, (
            f"feature 维度 {feature.shape[-1]} != {self._feature_dim}"
        )
        lead = feature.shape[:-1]   # 批次/时间维 (...,)

        return {
            # ---- 英雄块：[0:484] = frd(235) + emy(235) + main(14) ----
            "hero_frd":    feature[..., 0:235].reshape(*lead, 1, HERO_FRD_DIM),
            "hero_emy":    feature[..., 235:470].reshape(*lead, 1, HERO_EMY_DIM),
            "hero_main":   feature[..., 470:484],
            # ---- 小兵块：[484:628] = frd(4×18) + emy(4×18) ----
            "soldier_frd": feature[..., 484:556].reshape(*lead, SOLDIER_NUM, SOLDIER_DIM),
            "soldier_emy": feature[..., 556:628].reshape(*lead, SOLDIER_NUM, SOLDIER_DIM),
            # ---- 建筑块：[628:700] = frd(2×18) + emy(2×18) ----
            "organ_frd":   feature[..., 628:664].reshape(*lead, ORGAN_NUM, ORGAN_DIM),
            "organ_emy":   feature[..., 664:700].reshape(*lead, ORGAN_NUM, ORGAN_DIM),
            # ---- 全局信息：5 维 game_time + 20 维英雄身份 one-hot（官方 25）----
            "global":      feature[..., 700:725],
        }

    # -- 分支前向 ---------------------------------------------------------

    @staticmethod
    def _hero_branch(trunk: nn.Module, fc: nn.Module, x: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """英雄分支：返回 (进状态表示的完整 128 维, 进 target 头的 32 维)。

        x: (..., 1, 235) -> out: (..., 128)
        官方算法（algorithm_torch.py:255-273）：完整 128 维 fc 输出进 concat，
        ``split([96, 32])`` 只用于取后 32 维作为 target embedding。
        """
        h = trunk(x)                     # (..., 1, 256)
        h = fc(h)                        # (..., 1, 128)
        h = h.squeeze(-2)                # (..., 128)  单英雄，池化是恒等
        _, target_part = torch.split(
            h, [HERO_STATE_PART, HERO_TARGET_PART], dim=-1
        )
        return h, target_part

    @staticmethod
    def _unit_branch(trunk: nn.Module, fc: nn.Module, x: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """小兵/建筑分支：返回 (单位维 max-pool 后的 32 维, 全部单位的 (..., n, 32))。

        x: (..., n_units, unit_dim)
        max-pool 在单位维 (-2) 上进行。
        """
        h = trunk(x)                     # (..., n_units, 64)
        h = fc(h)                        # (..., n_units, 32)
        pooled = h.max(dim=-2).values    # (..., 32)
        return pooled, h

    # -- 主前向 -----------------------------------------------------------

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码 725 维特征。

        Args:
            feature: (..., 725)
        Returns:
            (state_embedding, target_units)
              state_embedding: (..., 512) 供 LSTM 使用
              target_units:    (..., 8, 32) 8 个 target 槽位的单位 embedding，
                               槽位语义见模块 docstring
        """
        b = self.split_feature(feature)

        # ---- 各分支 ----
        hero_main = self.hero_main_mlp(b["hero_main"])                       # (..., 16)

        hero_frd_state, hero_frd_tgt = self._hero_branch(
            self.hero_mlp, self.hero_frd_fc, b["hero_frd"])                  # (...,128) (...,32)
        hero_emy_state, hero_emy_tgt = self._hero_branch(
            self.hero_mlp, self.hero_emy_fc, b["hero_emy"])

        soldier_frd_pool, _ = self._unit_branch(
            self.soldier_mlp, self.soldier_frd_fc, b["soldier_frd"])         # (..., 32)
        soldier_emy_pool, soldier_emy_units = self._unit_branch(
            self.soldier_mlp, self.soldier_emy_fc, b["soldier_emy"])         # (...,32) (...,4,32)

        organ_frd_pool, _ = self._unit_branch(
            self.organ_mlp, self.organ_frd_fc, b["organ_frd"])               # (..., 32)
        organ_emy_pool, _ = self._unit_branch(
            self.organ_mlp, self.organ_emy_fc, b["organ_emy"])               # (..., 32)

        g = b["global"]                                                       # (..., 25)

        # ---- 状态表示 concat（顺序与官方 algorithm_torch.py:334-346 一致；
        #      因为后面接的是可学习 MLP，顺序其实可任意置换，
        #      这里保持与官方一致纯粹是为了便于逐层对标）----
        concat = torch.cat([
            soldier_frd_pool,     # 32
            soldier_emy_pool,     # 32
            organ_frd_pool,       # 32
            organ_emy_pool,       # 32
            hero_main,            # 16
            hero_frd_state,       # 128
            hero_emy_state,       # 128
            g,                    # 25
        ], dim=-1)                # -> (..., 425)

        state_embedding = self.head_mlp(concat)                              # (..., 512)

        # ---- 组装 8 个 target 槽位（顺序严格按官方，见模块 docstring）----
        lead = feature.shape[:-1]
        dummy = torch.full(
            (*lead, 1, HERO_TARGET_PART), 0.1,
            dtype=hero_emy_tgt.dtype, device=hero_emy_tgt.device,
        )                                                                    # idx 0：占位
        target_units = torch.cat([
            dummy,                                       # idx 0 无目标占位
            hero_emy_tgt.unsqueeze(-2),                  # idx 1 敌方英雄
            hero_frd_tgt.unsqueeze(-2),                  # idx 2 友方英雄
            soldier_emy_units,                           # idx 3-6 敌方小兵 ×4
            organ_emy_pool.unsqueeze(-2),                # idx 7 敌方建筑
        ], dim=-2)                                           # -> (..., 8, 32)

        return state_embedding, target_units


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """轻量多头 self-attention，Pre-LN + residual（本项目增强项，官方没有）。

    作用于 LSTM 输出 (B, T, d_model)。

    ────────────────────────────────────────────────────────────────────────
    ⚠️⚠️ 这个模块是"训练很差"的头号嫌疑，启用前务必读完 ──────────────────
    ────────────────────────────────────────────────────────────────────────
    它把**时间维**当作 attention 序列：

      * 采样侧（``env_runner.sample_action``）每次只喂 **T = 1** 的一帧，
        即 ``feat.reshape(1, 1, -1)``。此时 ``softmax`` 只有一个元素，
        attention **恒等于恒等映射** —— 这一帧根本看不到别的帧。
      * 训练侧（``learner.compute_loss``）喂的是 **(B, T=16)** 的 chunk，
        attention 在 16 帧上做加权聚合，每个时间步的输出都混入了
        同 chunk 其它帧的信息。

    也就是说 **old_log_probs 与 new_log_probs 来自两个不同的函数**：
    PPO 的 importance ratio（逐头 ``exp(logp_new_i - logp_old_i)``）
    在权重完全没更新时就已经偏离 1，KL 因此被"凭空"抬高并触发早停，
    clip 失去意义，value clip 的基准（old_values，采样侧算的）也和新值
    不在同一尺度上。历史症状（KL 硬停频繁 / minibatch 只用 1/8 /
    策略几乎不更新）与此完全吻合。

    更糟的是原实现**没有 causal mask**：训练时第 0 帧能看到第 1..15 帧，
    即**看到未来**。这等于给策略/价值网络开了信息泄漏的口子，
    训练指标会变好，但部署（T=1）时那部分信息并不存在。

    因此：
      1. 默认**关闭**（``NetworkConfig.use_time_attention = False``），
         回到官方 baseline 的 LSTM 结构；
      2. 若确实要启用，这里只提供 **causal** 版本（不泄漏未来），
         但采样侧仍必须喂足同样的时间上下文才能消除函数不一致，
         这一步本仓库尚未实现（见 network 报告），
         所以在 ``HoK1v1Network`` 里启用时会打 warning。

    Pre-LN 而非 Post-LN：训练更稳定，无需 warmup。
    """

    def __init__(self, d_model: int = 512, n_head: int = 4,
                 d_ff: int = 1024, dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} 不能被 n_head {n_head} 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # 注意力投影用更小的 gain，避免初期 attention 过度集中
        for m in [self.qkv, self.proj]:
            nn.init.orthogonal_(m.weight, gain=1.0 / math.sqrt(2.0))
            nn.init.constant_(m.bias, 0.0)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0 / math.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-LN Transformer block（自注意力 + FFN，均带残差）。

        Args:
            x: (B, T, d_model)。这里的 **T 是时间维**，attention 在时间上做。

        ⚠️ 采样侧每次只喂 T=1（单帧决策），此时 softmax 只有一个元素、
        该模块退化为恒等映射；训练侧喂 T=16 才会跨帧聚合。这正是历史上
        "old/new log prob 来自两个不同函数"的根源，所以
        ``use_time_attention`` 默认关闭（见模块顶部说明）。
        ``causal=True`` 时第 t 帧只看 0..t 帧，不泄漏未来。
        """
        # Pre-LN + residual
        h = self.ln1(x)
        B, T, _ = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)                       # (B, T, n_head, d_head)
        q = q.permute(0, 2, 1, 3)                         # (B, n_head, T, d_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal and T > 1:
            # 严格下三角：第 t 帧只能看 0..t 帧，绝不泄漏未来。
            # 用 dtype 的最小有限值而不是 -inf，避免 AMP fp16 下
            # (-inf) - (-inf) 产生 NaN。
            keep = torch.ones(T, T, dtype=torch.bool, device=attn.device).tril()
            attn = attn.masked_fill(~keep, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                    # (B, n_head, T, d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        x = x + self.drop(self.proj(out))
        # FFN block
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# 动作头 / Value 头
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """6 头动作输出。

    Heads 0-4：独立 logits（button / move_x / move_z / kill_x / kill_z）。

    Head 5 (target)：**基于单位 embedding 的点积**，而不是独立的 8 路 logits。
    这样 target 的分数天然来源于"这个单位长什么样"，与官方一致::

        logits_target = (target_embed_mlp(unit_emb) · query)   # (…, 8)

    其中 unit_emb 来自 FeatureEncoder 输出的 (…, 8, 32)。

    为什么不能用独立的 8 路 logits？
    因为 8 个槽位在不同时刻代表不同的单位（敌方英雄 / 4 个小兵 / 建筑），
    独立 logits 无法把"单位在当前观测中的样子"与打分关联起来，
    学不到"该打谁"的泛化能力。
    """

    def __init__(self, hidden_dim: int = 512, target_embed_dim: int = 32,
                 label_sizes: Optional[List[int]] = None):
        super().__init__()
        label_sizes = list(label_sizes or LABEL_SIZE_LIST)
        self.label_sizes = label_sizes

        # 5 个独立 label 头（button / move_x / move_z / kill_x / kill_z）
        self.label_mlps = nn.ModuleList([
            make_mlp([hidden_dim, 64, size]) for size in label_sizes[:5]
        ])

        # target 头：LSTM 隐状态 -> query；单位 embedding -> 投影后与 query 点积
        # 官方对应 lstm_tar_embed_mlp 与 target_embed_mlp(use_bias=False)
        self.tar_query = make_fc(hidden_dim, target_embed_dim)
        self.target_embed_mlp = make_fc(target_embed_dim, target_embed_dim, bias=False)

    def forward(self, hidden: torch.Tensor, target_units: torch.Tensor
                ) -> List[torch.Tensor]:
        """返回 6 个 logit tensor。

        Args:
            hidden:       (B, T, hidden_dim) LSTM + attention 后的隐状态
            target_units: (B, T, 8, target_embed_dim) 8 个 target 槽位的单位 embedding
        Returns:
            list，长度 6：
              [0..4] 形状 (B, T, size_i)
              [5]    形状 (B, T, 8)
        """
        logits_list = [mlp(hidden) for mlp in self.label_mlps]

        # query: (B, T, D) -> (B, T, D, 1)
        query = self.tar_query(hidden).unsqueeze(-1)
        # units: (B, T, 8, D) -> (B, T, 8, D)
        units = self.target_embed_mlp(target_units)
        # 点积 -> (B, T, 8, 1) -> (B, T, 8)
        target_logits = torch.matmul(units, query).squeeze(-1)
        logits_list.append(target_logits)

        return logits_list


class MultiHeadValue(nn.Module):
    """6 子头 value 头：hp / money / tower / kill / dead / win（本项目增强项）。

    官方只有一个 value 头（``value_mlp = MLP([512, 64, 1])``）。
    这里把 value 分解为 6 个子目标再按可学习权重混合，理由是
    dense reward 的各分量语义独立，分别预测可以加速信用分配。

    注意：当前 ``learner`` 只用组合后的 value 计算 value loss；
    ``value_heads`` 目前仅用于日志观测。若要让 6 个子头各自监督，
    需要为每个分量单独计算回报（Phase 3 TODO）。
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 6):
        super().__init__()
        self.n_heads = n_heads
        self.value_mlps = nn.ModuleList([
            make_mlp([hidden_dim, 64, 1]) for _ in range(n_heads)
        ])
        # 可学习混合权重（softmax 归一化，保证 value 量级稳定）
        self.mix_logits = nn.Parameter(torch.zeros(n_heads))
        self._head_names = ["hp", "money", "tower", "kill", "dead", "win"]

    def forward(self, hidden: torch.Tensor):
        """返回 (combined_value, head_values_dict)。"""
        head_values = {}
        head_tensors = []
        for i, mlp in enumerate(self.value_mlps):
            v = mlp(hidden).squeeze(-1)                    # (…,)
            head_values[self._head_names[i]] = v
            head_tensors.append(v)
        stacked = torch.stack(head_tensors, dim=-1)        # (…, n_heads)
        weights = F.softmax(self.mix_logits, dim=-1)
        combined = (stacked * weights).sum(dim=-1)         # (…,)
        return combined, head_values


# ---------------------------------------------------------------------------
# 顶层网络
# ---------------------------------------------------------------------------

class HoK1v1Network(nn.Module):
    """完整网络：encoder -> LSTM -> Attention -> ActionHead + MultiHeadValue。

    ⚠️ 关于 ``legal_action`` 参数：
    原实现的 ``forward`` 收了一个 ``legal_action`` 却从未使用（死参数），
    容易让人误以为掩码已经在网络内部完成。这里**直接移除该参数**，
    把掩码职责明确交给调用方（``env_runner`` 采样、``learner`` 算 loss），
    避免歧义。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_cfg = cfg.network

        self.encoder = FeatureEncoder(feature_dim=n_cfg.feature_dim)

        # LSTM：单层，hidden=512，与官方 baseline 完全一致
        self.lstm = nn.LSTM(
            input_size=n_cfg.lstm_hidden,
            hidden_size=n_cfg.lstm_hidden,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        # Self-Attention（本项目增强项，**默认关闭**）
        # 见 MultiHeadSelfAttention 的 docstring：它以时间维为 attention 序列，
        # 而采样侧每次只喂 T=1 的一帧，attention 恒等于恒等映射，
        # 导致 old_log_probs 与 new_log_probs 来自两个不同的函数。
        self.use_time_attention = bool(
            getattr(n_cfg, "use_time_attention", False)
        )
        self.attn = None
        if self.use_time_attention:
            logger.warning(
                "已启用 time-axis self-attention：采样侧 sample_action 只喂 T=1 的"
                "单帧，而训练侧喂 T=%d 的 chunk，两侧前向不是同一个函数，"
                "PPO 的 ratio / KL / value_clip 都会失真。"
                "除非同时改造采样侧喂入相同时间窗，否则请保持 "
                "use_time_attention=False。",
                n_cfg.lstm_time_steps,
            )
            self.attn = MultiHeadSelfAttention(
                d_model=n_cfg.lstm_hidden,
                n_head=n_cfg.attn_heads,
                d_ff=n_cfg.attn_d_ff,
                dropout=n_cfg.attn_dropout,
                causal=True,
            )

        self.action_head = ActionHead(
            hidden_dim=n_cfg.lstm_hidden,
            target_embed_dim=n_cfg.target_embed_dim,
            label_sizes=n_cfg.label_size_list,
        )
        self.value_head = MultiHeadValue(
            hidden_dim=n_cfg.lstm_hidden,
            n_heads=len(n_cfg.value_heads),
        )
        self.lstm_hidden = n_cfg.lstm_hidden

    def forward(self, feature: torch.Tensor,
                lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Dict[str, object]:
        """前向。

        Args:
            feature:    (B, T, 725)
            lstm_state: (h, c)，各 (1, B, hidden)；None 则零初始化
        Returns:
            dict:
              logits_list:    list[Tensor]，[0..4] 形状 (B,T,size_i)，
                              [5] 形状 (B,T,8)
              值域说明:       所有 logits 都是**未掩码**的原始输出，
                              掩码由调用方用 action_mask.masked_softmax 施加
              value:          (B, T) 组合 value
              value_heads:    dict name -> (B, T)
              lstm_state_new: (h, c)，各 (1, B, hidden)
        """
        B, T, _ = feature.shape

        emb, target_units = self.encoder(feature)          # (B,T,512) (B,T,8,32)

        if lstm_state is None:
            lstm_state = self.init_hidden(B, feature.device)

        lstm_out, lstm_state_new = self.lstm(emb, lstm_state)   # (B,T,hidden)
        if self.attn is not None:
            attn_out = self.attn(lstm_out)                      # (B,T,hidden)
        else:
            # 官方 baseline 结构：LSTM 输出直接进各 head。
            # 采样侧 T=1 与训练侧 T=16 在这里是**同一个函数**，
            # 这是 PPO ratio / KL / value_clip 成立的前提。
            attn_out = lstm_out

        logits_list = self.action_head(attn_out, target_units)
        value, value_heads = self.value_head(attn_out)

        return {
            "logits_list": logits_list,
            "value": value,
            "value_heads": value_heads,
            "lstm_state_new": lstm_state_new,
        }

    def init_hidden(self, batch_size: int, device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM 初始隐状态（零向量），形状 (1, B, hidden)。"""
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )

    def count_params(self) -> int:
        """统计可训练参数数量（用于日志）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
