#!/usr/bin/env bash
# 在 hok 环境中运行：bash scripts/run_tower_guard_ab.sh [control|candidate|both]
set -euo pipefail

cd "$(dirname "$0")/.."
stage="${1:-both}"
case "$stage" in
  control|candidate|both) ;;
  *) echo "用法: bash scripts/run_tower_guard_ab.sh [control|candidate|both]" >&2; exit 2 ;;
esac

checkpoint="runs/reward-events-control-stage32-v1/best_ckpt.pt"
expected_sha256="544f5db787371bc4f5213517aa98b318eadcbd04e9d2a997f30855dcdd58fb0b"
control_dir="runs/tower-guard-control-u24-8-v1"
candidate_dir="runs/tower-guard15000-u24-8-v1"
if [[ ! -f "$checkpoint" ]]; then
  echo "缺少更新 24 checkpoint: $checkpoint" >&2
  exit 1
fi
actual_sha256="$(sha256sum "$checkpoint" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "更新 24 checkpoint 指纹不符: $actual_sha256" >&2
  exit 1
fi
python - <<'PY'
from hok1v1_rl.config import load_config
control = load_config("configs/reward_events_stage8_control.yaml").to_dict()
candidate = load_config("configs/reward_events_tower_guard15000.yaml").to_dict()
differences = {(section, key) for section in control for key in control[section]
               if control[section][key] != candidate[section][key]}
if differences != {("action", "tower_guard_radius")}:
    raise SystemExit(f"塔前保护短测不是单因素变更: {differences}")
if control["action"]["tower_guard_radius"] != 0 or candidate["action"]["tower_guard_radius"] != 15000:
    raise SystemExit("塔前保护半径不符")
PY
if [[ "$stage" == control || "$stage" == both ]]; then
  if [[ -e "$control_dir" ]]; then
    echo "输出目录已存在: $control_dir" >&2
    exit 1
  fi
fi
if [[ "$stage" == candidate || "$stage" == both ]]; then
  if [[ -e "$candidate_dir" ]]; then
    echo "输出目录已存在: $candidate_dir" >&2
    exit 1
  fi
fi

if [[ "$stage" == control || "$stage" == both ]]; then
  python scripts/train_ppo.py \
    --config configs/reward_events_stage8_control.yaml \
    --init-from "$checkpoint" \
    --samplers 8 --sampler-device cpu --sampler-inference jit \
    --fragment-length 512 --burn-in 64 --device cuda:0 \
    --runtime-id hokguard-control-u24 --zmq-port 39700 \
    --total-steps 8 \
    --log-dir "$control_dir"
fi

if [[ "$stage" == candidate || "$stage" == both ]]; then
  python scripts/train_ppo.py \
    --config configs/reward_events_tower_guard15000.yaml \
    --init-from "$checkpoint" \
    --samplers 8 --sampler-device cpu --sampler-inference jit \
    --fragment-length 512 --burn-in 64 --device cuda:0 \
    --runtime-id hokguard-candidate-u24 --zmq-port 39740 \
    --total-steps 8 \
    --log-dir "$candidate_dir"
fi
