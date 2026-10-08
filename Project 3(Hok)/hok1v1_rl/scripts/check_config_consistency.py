# -*- coding: utf-8 -*-
"""校验 configs/default.yaml 与 config.py 默认值一致（P2 的防线）。

只需在项目根目录运行；不连 gamecore、不建网络。
退出码 0 = 一致，1 = 有漂移。
"""
import os
import sys

# run.sh 的 cwd 是项目根目录，但 cwd 不在 sys.path 里（只有脚本所在目录在），
# 所以显式把项目根目录加进去，脚本才能 import hok1v1_rl。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hok1v1_rl.config import Config
from hok1v1_rl.config import load_config

KEEP = [
    ("network.feature_dim", lambda c: c.network.feature_dim),
    ("network.use_hero_identity", lambda c: c.network.use_hero_identity),
    ("network.use_time_attention", lambda c: c.network.use_time_attention),
    ("network.attn_dropout", lambda c: c.network.attn_dropout),
    ("network.lstm_time_steps", lambda c: c.network.lstm_time_steps),
    ("train.amp", lambda c: c.train.amp),
    ("ppo.value_clip", lambda c: c.ppo.value_clip),
    ("ppo.normalize_returns", lambda c: c.ppo.normalize_returns),
    ("ppo.lr", lambda c: c.ppo.lr),
    ("ppo.lr_end", lambda c: c.ppo.lr_end),
    ("ppo.ppo_epochs", lambda c: c.ppo.ppo_epochs),
    ("ppo.batch_size", lambda c: c.ppo.batch_size),
    ("ppo.minibatch_size", lambda c: c.ppo.minibatch_size),
    ("ppo.entropy_beta_start", lambda c: c.ppo.entropy_beta_start),
    ("ppo.entropy_beta_end", lambda c: c.ppo.entropy_beta_end),
    ("ppo.gamma", lambda c: c.ppo.gamma),
    ("ppo.lamda", lambda c: c.ppo.lamda),
    ("ppo.clip_param", lambda c: c.ppo.clip_param),
    ("env.max_frame_num", lambda c: c.env.max_frame_num),
    ("env.predict_frequency", lambda c: c.env.predict_frequency),
    ("env.num_envs", lambda c: c.env.num_envs),
    ("env.reward_time_coef", lambda c: c.env.reward_time_coef),
    ("self_play.enabled", lambda c: c.self_play.enabled),
]

defaults = Config()
loaded = load_config("configs/default.yaml")

bad = 0
print("%-32s %-14s %-14s %s" % ("字段", "config.py默认", "yaml实际", "结果"))
print("-" * 78)
for name, get in KEEP:
    a, b = get(defaults), get(loaded)
    ok = (a == b)
    if not ok:
        bad += 1
    print("%-32s %-14s %-14s %s" % (name, a, b, "OK" if ok else "!! 漂移"))

print("-" * 78)
if bad:
    print("发现 %d 处漂移：config.py 与 default.yaml 不一致。" % bad)
    print("⚠️ 两者不一致时 YAML 会覆盖默认值，实际生效的是 YAML 那列。")
    sys.exit(1)
print("config.py 与 default.yaml 完全一致（%d 项）" % len(KEEP))

# 额外硬性判据
assert loaded.network.feature_dim == 725, "feature_dim 必须是 725（705 会缺英雄身份）"
assert loaded.train.amp is False, "amp 必须 false（否则两侧前向不一致）"
assert loaded.ppo.value_clip is False, "value_clip 必须 false（官方无此项）"
assert loaded.network.use_time_attention is False, "时间维 attention 必须关闭"
assert loaded.ppo.lr_end == loaded.ppo.lr, "不期望 LR 退火"
print("关键不变式全部满足（725 / amp=false / value_clip=false / 无 attention / LR 恒定）")
