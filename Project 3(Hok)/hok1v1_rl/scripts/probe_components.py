"""测量探针：跑完整一局，比较"旧 hp 口径"与"可见性安全的伤害口径"的量级。

背景（已实测）：
  * `hero.hp` 在英雄不可见时会变成占位值（实测为 1）。
    旧口径 `max(0, -Δhp)` 于是把"敌人脱离视野"记成"敌人挨了 ~3000 伤害"。
  * `hero.camp_visible` 是**按观察者**给出的可见性：
    对 camp C 的英雄，`camp_visible[j]` 表示"camp(j+1) 对 C 是否可见"。
  * hero 还有 `totalHurtToHero` / `totalBeHurtByHero` 累计计数器，与视野无关。

本脚本跑完整一局，同时累计多种口径，直接对比量级。

运行::

    bash run.sh scripts/probe_components.py --decisions 7000
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
log = logging.getLogger("probe_cmp")

ORGAN_UNIT = 12000.0
MONEY_UNIT = 1000.0
EXP_UNIT = 1000.0


def camp_of(x, default=-1):
    c = getattr(x, "camp", None)
    if c is None:
        return default
    try:
        return int(c)
    except (TypeError, ValueError):
        return int(getattr(c, "value", default))


def fld(x, name, default=0.0):
    try:
        return float(getattr(x, name))
    except (TypeError, ValueError, AttributeError):
        return default


def visible_to(observer_hero, camp, n_camps=2):
    """camp 阵营对 observer_hero 是否可见（camp_visible 按 camp 编号索引）。"""
    cv = getattr(observer_hero, "camp_visible", None)
    if cv is None:
        return True
    try:
        idx = int(camp) - 1
        if 0 <= idx < len(cv):
            return bool(cv[idx])
    except (TypeError, ValueError):
        pass
    return True


def random_legal_action(state):
    la = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)
    sp = split_legal_action_expanded(la)
    bs = [j for j in range(12) if float(sp[0][j]) == 1.0]
    b = int(random.choice(bs))
    tg = [j for j in range(8) if float(sp[5][b][j]) == 1.0]
    pick = lambda i: int(random.choice([j for j in range(16) if float(sp[i][j]) == 1.0]))
    return [b, pick(1), pick(2), pick(3), pick(4),
            int(random.choice(tg)) if tg else 0]


def organ_hp(req, camp):
    tot = 0.0
    for o in req.organ_list:
        if camp_of(o) == camp and int(getattr(o, "type", -1)) in (21, 24):
            tot += fld(o, "hp")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--decisions", type=int, default=7000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from hok1v1_rl.env_runner import build_hok_env, make_camp_config

    env = build_hok_env(cfg)
    camp_cfg = make_camp_config()
    use_common_ai = [False] + [True] * (cfg.env.player_num - 1)
    _o, _r, done, states = env.reset(camp_cfg, use_common_ai=use_common_ai, eval=False)
    pid = 0
    st = states[pid]

    my_rid = st["player_id"]
    my_camp = en_camp = None
    for h in st["req_pb"].hero_list:
        if getattr(h, "runtime_id", None) == my_rid:
            my_camp = camp_of(h)
    for h in st["req_pb"].hero_list:
        if camp_of(h) != my_camp:
            en_camp = camp_of(h)
    log.info("my_camp=%s en_camp=%s player_id=%r", my_camp, en_camp, my_rid)

    keys = ("hp_old", "hp_vis", "hp_ctr", "money_my", "money_diff",
            "exp_my", "exp_diff", "tower", "kill", "dead", "ep")
    acc = {k: 0.0 for k in keys}
    vis_loss = vis_gain = counter_back = 0
    prev = None
    last = None
    steps = 0

    while steps < args.decisions and not done[pid]:
        a6 = random_legal_action(st)
        actions = [tuple([0] * 6)] * cfg.env.player_num
        actions[pid] = tuple(int(x) for x in a6)
        _o, _r, done, states = env.step(actions)
        st = states[pid]
        steps += 1
        req = st.get("req_pb")
        if req is None:
            continue
        me = en = None
        for h in req.hero_list:
            if camp_of(h) == my_camp:
                me = h
            elif camp_of(h) == en_camp:
                en = h
        if me is None or en is None:
            continue

        cur = {
            "hp_me": fld(me, "hp"), "max_me": max(1.0, fld(me, "max_hp", 1.0)),
            "hp_en": fld(en, "hp"),
            "dealt": fld(me, "totalHurtToHero"), "taken": fld(me, "totalBeHurtByHero"),
            "money": fld(me, "money"), "money_en": fld(en, "money"),
            "exp": fld(me, "exp"), "exp_en": fld(en, "exp"),
            "ep": fld(me, "ep"),
            "kills": fld(me, "killCnt"), "deaths": fld(me, "deadCnt"),
            "organ_me": organ_hp(req, my_camp), "organ_en": organ_hp(req, en_camp),
            "vis_en": visible_to(me, en_camp),
            "vis_me": visible_to(me, my_camp),
        }
        if prev is not None:
            acc["hp_old"] += (max(0.0, -(cur["hp_en"] - prev["hp_en"]))
                              - max(0.0, -(cur["hp_me"] - prev["hp_me"])))
            if prev["vis_en"] and cur["vis_en"]:
                acc["hp_vis"] += (max(0.0, -(cur["hp_en"] - prev["hp_en"]))
                                  - max(0.0, -(cur["hp_me"] - prev["hp_me"])))
            d_dealt = cur["dealt"] - prev["dealt"]
            d_taken = cur["taken"] - prev["taken"]
            if d_dealt < -1.0 or d_taken < -1.0:
                counter_back += 1
            acc["hp_ctr"] += max(0.0, d_dealt) - max(0.0, d_taken)

            acc["money_my"] += cur["money"] - prev["money"]
            acc["money_diff"] += ((cur["money"] - prev["money"])
                                  - (cur["money_en"] - prev["money_en"]))
            acc["exp_my"] += cur["exp"] - prev["exp"]
            acc["exp_diff"] += ((cur["exp"] - prev["exp"])
                                - (cur["exp_en"] - prev["exp_en"]))
            acc["tower"] += ((cur["organ_me"] - prev["organ_me"])
                             - (cur["organ_en"] - prev["organ_en"]))
            acc["kill"] += cur["kills"] - prev["kills"]
            acc["dead"] += cur["deaths"] - prev["deaths"]
            acc["ep"] += cur["ep"] - prev["ep"]

            if prev["vis_en"] and not cur["vis_en"]:
                vis_loss += 1
            if (not prev["vis_en"]) and cur["vis_en"]:
                vis_gain += 1
        prev = cur
        last = cur
        if steps % 1000 == 0:
            log.info("  step %5d hp_old=%+10.0f hp_ctr=%+9.0f vis_en=%s hp_en=%.0f",
                     steps, acc["hp_old"], acc["hp_ctr"], cur["vis_en"], cur["hp_en"])

    log.info("=" * 70)
    log.info("对局结束：%d 个决策（约 %d 帧）done=%s", steps, steps * 3, done[pid])
    log.info("敌方可见性跳变：可见->不可见 %d 次，不可见->可见 %d 次", vis_loss, vis_gain)
    log.info("伤害计数器回退次数（负增量 > 1）: %d", counter_back)
    log.info("-" * 70)
    for k in keys:
        log.info("    %-11s = %+14.1f", k, acc[k])
    log.info("-" * 70)
    if last is not None:
        mh = last["max_me"]
        log.info("归一化单位（除以基准）：")
        log.info("    hp_old /maxhp(%d) = %+.2f 个血条", int(mh), acc["hp_old"] / mh)
        log.info("    hp_vis /maxhp     = %+.2f", acc["hp_vis"] / mh)
        log.info("    hp_ctr /maxhp     = %+.2f", acc["hp_ctr"] / mh)
        log.info("    tower  /12000     = %+.2f 座基地", acc["tower"] / ORGAN_UNIT)
        log.info("    money_diff /1000  = %+.2f", acc["money_diff"] / MONEY_UNIT)
        log.info("    exp_diff   /1000  = %+.2f", acc["exp_diff"] / EXP_UNIT)
        log.info("    kill=%+.1f dead=%+.1f", acc["kill"], acc["dead"])

    try:
        env.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
