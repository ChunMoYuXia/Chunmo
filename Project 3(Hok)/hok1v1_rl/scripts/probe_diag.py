"""诊断探针：一次性回答几个"必须实测"的问题。

1. `observation` 真实维度是多少（705 or 725）？
2. 原始 reward 向量多长？哪一位是 `reward_sum`（= 其余分量之和）？
3. `req_pb` 里英雄的 `hp` / `max_hp` / `killCnt` / `deadCnt` 量级是多少？
   （训练奖励的 hp 项是"每局平均十几万"，需要确认这是不是单位问题）
4. 建筑（塔/水晶）的 hp / max_hp / type 是多少？
5. `legal_action` 前 5 段的 index 0 到底是不是**永远非法**
   （训练日志里 6 个头的 index 0 命中率恒为 0.0%，需要确认是策略选择还是 mask）
6. `sub_action_mask` 的实际内容（官方的 head 屏蔽权重）

运行（必须在项目根目录，且 Windows 侧 gamecore 已启动）::

    conda activate hok
    python scripts/probe_diag.py --decisions 400
"""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hok1v1_rl.config import load_config
from hok1v1_rl.action_mask import split_legal_action_expanded

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("probe_diag")


def show_attrs(obj, label, only=None):
    out = []
    for a in sorted(x for x in dir(obj) if not x.startswith("_")):
        if only is not None and a not in only:
            continue
        try:
            v = getattr(obj, a)
        except Exception:
            continue
        if callable(v):
            continue
        out.append(f"{a}={v!r}")
    log.info("  %s: %s", label, "  ".join(out) if out else "(无)")


def random_legal_action(state):
    la = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)
    sp = split_legal_action_expanded(la)
    buttons = [j for j in range(12) if float(sp[0][j]) == 1.0]
    b = int(random.choice(buttons))
    tg = [j for j in range(8) if float(sp[5][b][j]) == 1.0]
    pick = lambda i, n: int(random.choice([j for j in range(n) if float(sp[i][j]) == 1.0]))
    return [b, pick(1, 16), pick(2, 16), pick(3, 16), pick(4, 16),
            int(random.choice(tg)) if tg else 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--decisions", type=int, default=400)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from hok1v1_rl.env_runner import build_hok_env, make_camp_config

    env = build_hok_env(cfg)
    camp = make_camp_config()
    use_common_ai = [False] + [True] * (cfg.env.player_num - 1)
    _obs, reward, done, states = env.reset(camp, use_common_ai=use_common_ai, eval=False)
    pid = 0
    st = states[pid]

    log.info("=" * 78)
    log.info("【1】state 的键: %s", sorted(st.keys()))
    log.info("【1】observation 长度 = %d", len(np.asarray(st["observation"]).reshape(-1)))
    log.info("【1】legal_action 长度 = %d", len(np.asarray(st["legal_action"]).reshape(-1)))
    log.info("【1】camp_config = %s", camp)
    log.info("【1】player_id = %r", st.get("player_id"))

    log.info("=" * 78)
    log.info("【6】sub_action_mask（官方 head 权重）:")
    sm = st.get("sub_action_mask")
    if isinstance(sm, dict):
        for k in sorted(sm, key=lambda x: int(x)):
            log.info("    button %-3s -> %s", k, np.asarray(sm[k]).tolist())
    else:
        log.info("    %r", sm)

    log.info("=" * 78)
    log.info("【5】legal_action 各段 index 0 是否合法 + 该段合法个数（首帧）")
    la = np.asarray(st["legal_action"], dtype=np.float32).reshape(-1)
    sp = split_legal_action_expanded(la)
    for i in range(5):
        v = np.asarray(sp[i]).reshape(-1)
        log.info("    head_%d dim=%d idx0=%.0f 合法数=%d  全文=%s",
                 i, v.shape[0], v[0], int((v == 1).sum()), v.tolist())
    t2 = np.asarray(sp[5])
    log.info("    head_5 (12x8) idx[button,0] 列: %s", t2[:, 0].tolist())
    log.info("    head_5 每行合法数: %s", [int((t2[b] == 1).sum()) for b in range(12)])

    log.info("=" * 78)
    log.info("【2】reward 向量")
    raw = np.asarray(st["reward"], dtype=np.float64).reshape(-1)
    ret = np.asarray(reward[pid], dtype=np.float64).reshape(-1)
    log.info("    state['reward'] 长度=%d 值=%s", raw.shape[0], raw.tolist())
    log.info("    env.step 返回的 reward[pid] 长度=%d 值=%s", ret.shape[0], ret.tolist())
    for k in range(raw.shape[0]):
        others = raw[[i for i in range(raw.shape[0]) if i != k]]
        if np.allclose(raw[k], others.sum(), rtol=1e-3, atol=1e-2):
            log.info("    >>> raw[%d] ≈ 其余分量之和 => 极可能是 reward_sum", k)
    log.info("    sum(raw)=%.4f  sum(ret)=%.4f", raw.sum(), ret.sum())

    log.info("=" * 78)
    log.info("【3】【4】req_pb 结构（首帧）")
    req = st.get("req_pb")
    if req is None:
        log.info("    req_pb 为 None（首帧可能还没有）")
    else:
        log.info("  hero_list 个数 = %d", len(req.hero_list))
        for i, h in enumerate(req.hero_list):
            show_attrs(h, f"hero[{i}]")
        log.info("  organ_list 个数 = %d", len(req.organ_list))
        for i, o in enumerate(req.organ_list):
            show_attrs(o, f"organ[{i}]")

    # ---- 步进采样：观察 hp 量级、index0 是否出现过合法/被采样 ----
    log.info("=" * 78)
    log.info("步进 %d 个决策，观察 hp 量级与 index0 合法性 ...", args.decisions)
    idx0_legal = np.zeros(6, dtype=np.int64)
    idx0_chosen = np.zeros(6, dtype=np.int64)
    frames = 0
    hp_records = []
    diag_hp = []          # (my_hp, my_max, en_hp, en_max, my_kill, my_dead)
    raw_sum_hits = 0
    reward_sum_idx = None

    while frames < args.decisions and not done[pid]:
        a6 = random_legal_action(st)
        actions = [tuple([0] * 6)] * cfg.env.player_num
        actions[pid] = tuple(int(x) for x in a6)
        _o, reward, done, states = env.step(actions)
        st = states[pid]
        frames += 1

        la = np.asarray(st["legal_action"], dtype=np.float32).reshape(-1)
        sp = split_legal_action_expanded(la)
        for i in range(5):
            if float(np.asarray(sp[i]).reshape(-1)[0]) == 1.0:
                idx0_legal[i] += 1
        if float(np.asarray(sp[5])[a6[0], 0]) == 1.0:
            idx0_legal[5] += 1
        for i in range(6):
            if a6[i] == 0:
                idx0_chosen[i] += 1

        raw = np.asarray(st["reward"], dtype=np.float64).reshape(-1)
        if reward_sum_idx is None:
            for k in range(raw.shape[0]):
                others = raw[[i for i in range(raw.shape[0]) if i != k]]
                if np.allclose(raw[k], others.sum(), rtol=1e-3, atol=1e-2):
                    reward_sum_idx = k
                    log.info("    >>> 步进中也命中：raw[%d] ≈ 其余之和", k)
                    break
        elif np.allclose(raw[reward_sum_idx],
                         raw[[i for i in range(raw.shape[0]) if i != reward_sum_idx]].sum(),
                         rtol=1e-3, atol=1e-2):
            raw_sum_hits += 1

        req = st.get("req_pb")
        if req is not None and len(hp_records) < 4000:
            for h in req.hero_list:
                hp_records.append((int(h.camp), float(h.hp)))
            if frames % 40 == 0:
                desc = []
                for h in req.hero_list:
                    desc.append(
                        f"camp{int(h.camp)} hp={float(h.hp):.1f} "
                        f"max_hp={getattr(h, 'max_hp', 'NA')} "
                        f"kill={getattr(h, 'killCnt', 'NA')} "
                        f"dead={getattr(h, 'deadCnt', 'NA')}"
                    )
                log.info("    f%-5d %s", frames, " | ".join(desc))
                if req.organ_list:
                    log.info("           organ: %s", [
                        f"camp{int(o.camp)} type={int(getattr(o,'type',-1))} "
                        f"hp={float(o.hp):.0f} max={getattr(o,'max_hp','NA')}"
                        for o in req.organ_list
                    ])

    log.info("=" * 78)
    log.info("【5】结论：index 0 的合法性 / 被采样次数")
    for i in range(6):
        log.info("    head_%d: index0 合法帧数=%d / %d ; 随机采样到 index0 次数=%d",
                 i, idx0_legal[i], frames, idx0_chosen[i])

    log.info("=" * 78)
    log.info("【3】英雄 hp 统计（全部采样帧）")
    if hp_records:
        camps = sorted({c for c, _ in hp_records})
        for c in camps:
            vals = np.array([v for cc, v in hp_records if cc == c])
            log.info("    camp %d: n=%d hp min=%.1f p50=%.1f max=%.1f",
                     c, vals.size, vals.min(), np.median(vals), vals.max())

    log.info("=" * 78)
    log.info("【2】reward_sum 判定：索引=%s，命中帧数=%d / %d",
             reward_sum_idx, raw_sum_hits, frames)

    try:
        env.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
