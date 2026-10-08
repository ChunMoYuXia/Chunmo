# -*- coding: utf-8 -*-
"""对照实验探针：把"胜负判定"与"推塔差分"放在一起对账。

要回答的两个问题（都来自诊断文档 §6）：
  1. `determine_outcome`（胜负）和 `state_reward_terms` 的 `tower` 差分
     是否**互相自洽**？历史日志里出现过"win_rate=1 与 tower=丢整座基地同时成立"。
  2. 阵营映射（`_resolve_camps`）对不对？它是两者的共同上游，错了会一起错。

做法：用**随机策略**跑若干局（策略不代表训练中期的行为，但足以让对局
自然结束、产生胜负），每局打印：
  * `_resolve_camps` 解析出的 my/en camp、player_id、runtime_id
  * 终局帧双方水晶/塔血量、归零建筑清单
  * `determine_outcome` 的返回值、据此发放的终端奖励
  * 全轨迹的**推塔差分**（来自 tower 分量的逐帧原始值）与最深掉血时刻
判据：
  * outcome == "win"  ⇒ 敌方水晶应归零
  * outcome == "lose" ⇒ 我方水晶应归零
  * outcome == "win" 与 tower 差分恒负**不应同时出现**

运行（Windows 侧 gamecore 已启动）::

    bash run.sh scripts/probe_outcome.py --episodes 3 --decisions 9000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from hok1v1_rl.config import load_config
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.reward import DEFAULT_WEIGHTS

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("probe_outcome")

ORGAN_UNIT = 12000.0     # tower 分量的归一化单位（1.0 = 一座基地）
TOWER_W = DEFAULT_WEIGHTS.tower


def fnum(v, nd=1):
    """格式化可能为 None 的数值（None 原样显示为 "None"）。"""
    if v is None:
        return "None"
    try:
        return ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return str(v)


def main():
    """跑若干整局并逐局对账，最后打印汇总表（退出码恒为 0，看输出判断）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=9000,
                    help="单局最大决策数（随机策略下对局很长，给足空间）")
    ap.add_argument("--max-frame-num", type=int, default=40000,
                    help="临时提高 max_frame_num，避免随机策略下频繁截断")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    # 随机策略的对局远长于训练策略（实测自然结束约 2100~3400 决策，
    # 随机策略可能翻几倍），这里临时放宽上限，减少"截断掩盖判定"的情况。
    cfg.env.max_frame_num = int(args.max_frame_num)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("probe_outcome：胜负判定 vs 推塔差分 对账")
    print("  episodes=%d decisions=%d max_frame_num=%d device=%s"
          % (args.episodes, args.decisions, cfg.env.max_frame_num, device))
    print("  tower 权重=%.1f  ORGAN_UNIT=%.0f（1.0 = 一座基地）" % (TOWER_W, ORGAN_UNIT))
    print("=" * 100)

    network = HoK1v1Network(cfg).to(device)
    env = build_hok_env(cfg, runtime_id=cfg.env.runtime_id)
    runner = EnvRunner(env, network, cfg, device)

    rows = []
    for ep in range(args.episodes):
        try:
            traj = runner.collect_trajectory(
                max_decisions=args.decisions, random_policy=True)
        except Exception as exc:              # noqa: BLE001
            print("\n[episode %d] collect_trajectory 失败：%s" % (ep, exc))
            import traceback
            traceback.print_exc()
            break

        es = traj["episode_stats"]
        ev = es.get("outcome_evidence") or {}
        outcome = ev.get("outcome")
        win = bool(traj["win"])
        trunc = bool(es.get("truncated", False))

        # ---- 逐帧推塔差分（原始单位）----
        # rews 的最后一个有效步含终端 ±50，必须排除，否则会被误读成"推塔"
        rews = np.asarray(traj["rewards"]).reshape(-1)
        n_valid = int(np.asarray(traj["valid_mask"]).sum())
        steps = rews[:max(0, n_valid - 1)]
        tower_norm = steps / TOWER_W if TOWER_W else steps
        tower_raw = tower_norm * ORGAN_UNIT
        tower_sum = float(tower_raw.sum())
        worst_i = int(np.argmin(tower_raw)) if tower_raw.size else -1
        worst_v = float(tower_raw[worst_i]) if worst_i >= 0 else 0.0

        # ---- 自洽性判据 ----
        checks = []
        if trunc:
            checks.append(("TRUNCATED(判定不参与对账)", False))
        else:
            if outcome == "win":
                checks.append(("win ⇒ 敌方水晶归零",
                               ev.get("crystal_en") is not None
                               and float(ev["crystal_en"]) <= 0.0))
            elif outcome == "lose":
                checks.append(("lose ⇒ 我方水晶归零",
                               ev.get("crystal_my") is not None
                               and float(ev["crystal_my"]) <= 0.0))
            elif outcome is None:
                checks.append(("outcome=None ⇒ 不发终端奖励（合法兜底）", True))
            # tower 差分与胜负必须同号
            if outcome == "win":
                checks.append(("win ⇒ 推塔差分 > 0", tower_sum > 0))
            elif outcome == "lose":
                checks.append(("lose ⇒ 推塔差分 < 0", tower_sum < 0))

        print()
        print("-" * 100)
        print("[episode %d] outcome=%s  win=%s  truncated=%s  决策数=%d  "
              "chunk 补齐=%d  终局步入 batch=%s"
              % (ep, outcome, win, trunc, es.get("episode_len"),
                 es.get("chunk_padded_steps"), es.get("last_step_valid")))
        print("-" * 100)
        print("  阵营映射: player_id=%s my_runtime_id=%s  my_camp=%s  en_camp=%s"
              % (ev.get("player_id"), ev.get("my_runtime_id"),
                 ev.get("my_camp"), ev.get("en_camp")))
        print("  hero_list: %s" % (ev.get("hero_camps"),))
        print("  终局建筑: 共 %s 个 | 水晶 my=%s en=%s | 塔 my=%s en=%s"
              % (ev.get("organ_count"),
                 fnum(ev.get("crystal_my")), fnum(ev.get("crystal_en")),
                 fnum(ev.get("tower_my")), fnum(ev.get("tower_en"))))
        print("  归零建筑: 水晶 camp=%s  塔 camp=%s"
              % (ev.get("crystal_zero_camps"), ev.get("tower_zero_camps")))
        print("  建筑总血量: my=%s  en=%s"
              % (fnum(ev.get("organ_total_my")), fnum(ev.get("organ_total_en"))))
        print("  推塔差分: 累计=%s（%+.3f 座基地）  最深掉血=%s @step %d"
              % (fnum(tower_sum, 0), tower_sum / ORGAN_UNIT,
                 fnum(worst_v, 0), worst_i))
        print("  奖励分量（每局均值）: %s"
              % {k: round(float(v), 3)
                 for k, v in (es.get("reward_components") or {}).items()})
        ok_all = True
        for name, ok in checks:
            ok_all = ok_all and ok
            print("    [%s] %s" % ("OK" if ok else "!!", name))
        rows.append((ep, outcome, win, trunc, tower_sum, ok_all))

    print()
    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print("%-4s %-8s %-6s %-10s %-16s %s"
          % ("ep", "outcome", "win", "truncated", "tower差分(座)", "自洽"))
    bad = 0
    for ep, outcome, win, trunc, tsum, ok in rows:
        if not ok:
            bad += 1
        print("%-4d %-8s %-6s %-10s %+16.3f %s"
              % (ep, outcome, win, trunc, tsum / ORGAN_UNIT,
                 "OK" if ok else "!! 不自洽"))
    print()
    if not rows:
        print("没有完成任何一局，无法对账。")
    elif bad == 0:
        print("全部 %d 局自洽：determine_outcome 与 tower 差分方向一致。" % len(rows))
    else:
        print("%d / %d 局不自洽 —— 需要进一步定位（先看是否 truncated）。"
              % (bad, len(rows)))

    try:
        runner.close()
    except Exception:      # noqa: BLE001
        pass
    try:
        env.close_game(force=True)
    except Exception:      # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
