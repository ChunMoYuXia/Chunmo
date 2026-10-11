#!/usr/bin/env bash
# 在 hok 环境中运行：bash scripts/run_tower_survival_ab.sh [control|candidate|both]
set -euo pipefail

cd "$(dirname "$0")/.."
stage="${1:-both}"
case "$stage" in
  control|candidate|both) ;;
  *) echo "用法: bash scripts/run_tower_survival_ab.sh [control|candidate|both]" >&2; exit 2 ;;
esac

checkpoint="runs/reward-events-control-stage32-v1/best_ckpt.pt"
if [[ ! -f "$checkpoint" ]]; then
  echo "缺少更新 24 checkpoint: $checkpoint" >&2
  exit 1
fi

if [[ "$stage" == control || "$stage" == both ]]; then
  python scripts/train_ppo.py \
    --config configs/reward_events_stage8_control.yaml \
    --init-from "$checkpoint" \
    --samplers 8 --sampler-device cpu --sampler-inference jit \
    --fragment-length 512 --burn-in 64 --device cuda:0 \
    --runtime-id hoktower-survival-control --zmq-port 38600 \
    --total-steps 8 \
    --log-dir runs/tower-survival-control-u24-8-v1
fi

if [[ "$stage" == candidate || "$stage" == both ]]; then
  python scripts/train_ppo.py \
    --config configs/reward_events_dead4.yaml \
    --init-from "$checkpoint" --allow-reward-change \
    --samplers 8 --sampler-device cpu --sampler-inference jit \
    --fragment-length 512 --burn-in 64 --device cuda:0 \
    --runtime-id hoktower-survival-dead4 --zmq-port 38640 \
    --total-steps 8 \
    --log-dir runs/tower-survival-dead4-u24-8-v1
fi
