#!/usr/bin/env bash
# 在 hok 环境中运行：bash scripts/run_economy2_short.sh
# 使用已完成的 runs/lr-control-u24-8-v1 作为 1e-4、economy_diff=0.5 对照。
set -euo pipefail

cd "$(dirname "$0")/.."
checkpoint="runs/reward-events-control-stage32-v1/best_ckpt.pt"
expected_sha256="544f5db787371bc4f5213517aa98b318eadcbd04e9d2a997f30855dcdd58fb0b"
control_status="runs/lr-control-u24-8-v1/status.json"
candidate_dir="runs/economy2-u24-8-v1"

if [[ ! -f "$checkpoint" ]]; then
  echo "缺少更新 24 checkpoint: $checkpoint" >&2
  exit 1
fi
actual_sha256="$(sha256sum "$checkpoint" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "更新 24 checkpoint 指纹不符: $actual_sha256" >&2
  exit 1
fi
if [[ ! -f "$control_status" ]]; then
  echo "缺少已完成的学习率对照: $control_status" >&2
  exit 1
fi
python - "$control_status" "$checkpoint" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from hok1v1_rl.config import load_config

status = json.load(open(sys.argv[1], encoding="utf-8"))
if status.get("phase") != "finished" or status.get("completed_updates") != 8:
    raise SystemExit("学习率 1e-4 对照未完成 8 次更新")
manifest = json.load(open("runs/lr-control-u24-8-v1/manifest.json", encoding="utf-8"))
if manifest["initialization"]["source_checkpoint"] != str(Path(sys.argv[2]).resolve()):
    raise SystemExit("对照来源 checkpoint 不一致")
if manifest["config"]["ppo"]["lr"] != 1e-4 or manifest["config"]["reward"]["economy_diff"] != 0.5:
    raise SystemExit("对照训练参数不符")
for source, expected in manifest["source_sha256"].items():
    path = Path(source)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"对照后源码发生变化: {source}")
control = load_config("configs/reward_events_stage8_control.yaml").to_dict()
candidate = load_config("configs/reward_events_economy2.yaml").to_dict()
differences = {(section, key) for section in control for key in control[section]
               if control[section][key] != candidate[section][key]}
if differences != {("reward", "economy_diff")} or candidate["reward"]["economy_diff"] != 2.0:
    raise SystemExit(f"候选配置不是经济差奖励单因素变更: {differences}")
PY
if [[ -e "$candidate_dir" ]]; then
  echo "输出目录已存在: $candidate_dir" >&2
  exit 1
fi

python scripts/train_ppo.py \
  --config configs/reward_events_economy2.yaml \
  --init-from "$checkpoint" --allow-reward-change \
  --samplers 8 --sampler-device cpu --sampler-inference jit \
  --fragment-length 512 --burn-in 64 --device cuda:0 \
  --runtime-id hokeconomy2-u24 --zmq-port 39440 \
  --total-steps 8 \
  --log-dir "$candidate_dir"
