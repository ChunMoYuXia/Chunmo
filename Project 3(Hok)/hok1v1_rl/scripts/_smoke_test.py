"""快速冒烟测试 —— 不依赖 gamecore，验证配置 / 动作空间 / 网络 / 损失 的通路。

运行::

    conda activate hok
    python scripts/_smoke_test.py

⚠️ 原版本的断言 ``illegal < 1e-6`` 在数学上不可能通过：
原 masked_softmax 写成 ``mask * exp(x) + eps``（eps 在乘 mask 之后），
非法位置会各自残留 1e-5 量级的概率，12 头里 9 个非法时总和约 5e-5。
现在 masked_softmax 已改为官方的 ``(exp(x) + eps) * mask``，非法概率**严格为 0**，
因此断言改为 ``== 0``。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from hok1v1_rl.config import load_config
from hok1v1_rl.action_mask import (
    LABEL_SIZE_LIST,
    LEGAL_ACTION_TOTAL_COMPRESSED,
    LEGAL_ACTION_TOTAL_EXPANDED,
    compress_legal_action,
    masked_softmax,
    split_legal_action_compressed,
    split_legal_action_expanded,
)
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.reward import compute_dense_reward, compute_value_targets, parse_env_reward
from hok1v1_rl.vram_guard import VRamGuard

# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
cfg = load_config(str(cfg_path))
print(f"[OK] config: num_envs={cfg.env.num_envs} batch={cfg.ppo.batch_size} "
      f"minibatch={cfg.ppo.minibatch_size} seq={cfg.network.lstm_time_steps}")
print(f"[OK] LABEL_SIZE_LIST={LABEL_SIZE_LIST}")
print(f"[OK] legal_action: expanded={LEGAL_ACTION_TOTAL_EXPANDED} "
      f"compressed={LEGAL_ACTION_TOTAL_COMPRESSED}")
assert sum(LABEL_SIZE_LIST) == LEGAL_ACTION_TOTAL_COMPRESSED
assert sum(LABEL_SIZE_LIST[:-1]) + LABEL_SIZE_LIST[0] * LABEL_SIZE_LIST[-1] == LEGAL_ACTION_TOTAL_EXPANDED

# ---------------------------------------------------------------------------
# 2. masked_softmax 语义（非法概率必须严格为 0）
# ---------------------------------------------------------------------------
logits = torch.randn(12)
mask = torch.tensor([1., 1., 1.] + [0.] * 9)
probs = masked_softmax(logits, mask)
illegal = float((probs * (1 - mask)).sum().item())
legal_sum = float((probs * mask).sum().item())
print(f"[OK] masked_softmax: 非法概率总和={illegal:.2e}（官方语义下应为 0）")
assert illegal == 0.0, f"mask 泄漏：{illegal}"
assert abs(legal_sum - 1.0) < 1e-6, f"合法概率和 {legal_sum} != 1"

# 全零 mask 的兜底：不应产生 NaN
allzero = masked_softmax(torch.randn(12), torch.zeros(12))
assert torch.isfinite(allzero).all(), "全零 mask 产生了非有限值"
print(f"[OK] 全零 mask 兜底：probs 有限，sum={float(allzero.sum()):.4f}")

# ---------------------------------------------------------------------------
# 3. 172 -> 84 压缩
# ---------------------------------------------------------------------------
la172 = (torch.rand(172) > 0.5).float()
splits = split_legal_action_expanded(la172)
button = 3
la84 = compress_legal_action(la172, torch.tensor(button))
assert la84.shape[0] == LEGAL_ACTION_TOTAL_COMPRESSED
assert torch.equal(la84[76:84], splits[5][button]), "172→84 的 target 行选错"
print(f"[OK] 172→84 压缩：button={button} 的 target mask={la84[76:84].tolist()}")

# 84 维切分回来应与压缩前一致
back = split_legal_action_compressed(la84)
assert back[-1].shape[0] == 8
assert torch.equal(back[-1], splits[5][button])
print("[OK] 84 维切分自洽")

# ---------------------------------------------------------------------------
# 4. 奖励与 GAE
# ---------------------------------------------------------------------------
fake_reward = [0.5, 0.3, -0.2, 0.1, 0.0, 0.0, -0.4, 0.2]
dense, comp = compute_dense_reward(fake_reward)
print(f"[OK] dense reward={dense:.4f}  hp={comp['hp']} money={comp['money']} tower={comp['tower']}")
# parse_env_reward 内部转 float32，float32(-0.4) 有 1e-8 级舍入，
# 精确等值断言恒失败，用容差比较
assert abs(parse_env_reward(fake_reward)["tower"] + 0.4) < 1e-6

# GAE：验证 done 掩码确实切断跨 episode 的 bootstrap
adv_nodone, _ = compute_value_targets(
    rewards=np.array([1.0, 1.0, 1.0], np.float32), gamma=0.99,
    values=np.array([5.0, 5.0, 5.0], np.float32), last_value=100.0,
    lamda=0.95, dones=np.array([0.0, 0.0, 0.0], np.float32),
)
adv_done, _ = compute_value_targets(
    rewards=np.array([1.0, 1.0, 1.0], np.float32), gamma=0.99,
    values=np.array([5.0, 5.0, 5.0], np.float32), last_value=100.0,
    lamda=0.95, dones=np.array([0.0, 0.0, 1.0], np.float32),
)
print(f"[OK] GAE 无终止: adv[-1]={adv_nodone[-1]:.3f}（含 bootstrap）")
print(f"[OK] GAE 有终止: adv[-1]={adv_done[-1]:.3f}（已切断，应等于 r-V=-4.0）")
assert abs(adv_done[-1] - (-4.0)) < 1e-4, f"done 掩码未生效：{adv_done[-1]}"
assert adv_nodone[-1] > 90.0, "未终止时应当 bootstrap last_value"

# ---------------------------------------------------------------------------
# 5. 网络前向（含 target 头的 8 槽位）
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}")
net = HoK1v1Network(cfg).to(device)
print(f"[OK] 网络参数量: {net.count_params() / 1e6:.2f}M")

B, T = 2, cfg.network.lstm_time_steps
feat = torch.randn(B, T, cfg.network.feature_dim, device=device)
out = net(feat, None)
assert len(out["logits_list"]) == 6, "应有 6 个动作头"
for i, lg in enumerate(out["logits_list"]):
    assert lg.shape == (B, T, LABEL_SIZE_LIST[i]), (
        f"head {i} 形状 {tuple(lg.shape)} != {(B, T, LABEL_SIZE_LIST[i])}"
    )
print(f"[OK] 6 个动作头形状: {[tuple(l.shape) for l in out['logits_list']]}")
print(f"[OK] value 形状={tuple(out['value'].shape)}  "
      f"value_heads={list(out['value_heads'].keys())}")
# target 头的 logits 不应恒为常数（说明确实依赖单位 embedding 与 query）
tgt = out["logits_list"][5]
assert tgt.std().item() > 0, "target logits 无变化，target 头可能退化为占位实现"
print(f"[OK] target logits 标准差={tgt.std().item():.4f}（非常数，说明点积生效）")

# ---------------------------------------------------------------------------
# 6. 特征切分顺序（防止 hero 块错位回归）
# ---------------------------------------------------------------------------
enc = net.encoder
probe = torch.zeros(1, 1, cfg.network.feature_dim)
probe[..., 0] = 1.0            # hero_frd 的第 0 维
probe[..., 470] = 2.0          # hero_main 的第 0 维
probe[..., 484] = 3.0          # soldier_frd 的第 0 维
probe[..., 700] = 4.0          # global 的第 0 维（game_time 起点）
probe[..., 705] = 5.0          # 英雄身份 one-hot 的第 0 维
sp = enc.split_feature(probe)
assert sp["hero_frd"].reshape(-1)[0] == 1.0, "hero_frd 切分起点错误（应为 0:235）"
assert sp["hero_main"].reshape(-1)[0] == 2.0, "hero_main 切分起点错误（应为 470:484）"
assert sp["soldier_frd"].reshape(-1)[0] == 3.0, "soldier_frd 切分起点错误（应为 484）"
assert sp["global"].reshape(-1)[0] == 4.0, "global 切分起点错误（应为 700）"
# global = 5 维 game_time + 20 维英雄身份 one-hot = 25（官方 DIM_OF_GLOBAL_INFO）
assert sp["global"].shape[-1] == 25, (
    f"global 应为 25 维（5 game_time + 20 英雄身份），实际 {sp['global'].shape[-1]}；"
    "若这里是 5，说明 feature_dim 又退回 705，英雄身份 one-hot 丢了"
)
assert sp["global"].reshape(-1)[5] == 5.0, "英雄身份 one-hot 切分起点错误（应为 705）"
print(f"[OK] 特征切分顺序正确: hero_frd=0:235, hero_emy=235:470, hero_main=470:484, "
      f"global=700:{cfg.network.feature_dim}（{sp['global'].shape[-1]} 维）")

# ---------------------------------------------------------------------------
# 7. VRamGuard（CPU 安全）
# ---------------------------------------------------------------------------
guard = VRamGuard(warn_gb=9.0, degrade_gb=10.5, oom_gb=11.5)
status = guard.check(step=0)
print(f"[OK] vram_status: used={status.used_gb:.2f}G reserved={status.reserved_gb:.2f}G "
      f"degraded={status.degraded}")

print("\n=== 所有冒烟测试通过 ===")
