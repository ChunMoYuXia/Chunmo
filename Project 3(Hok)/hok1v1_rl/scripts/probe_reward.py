"""奖励分量语义探针 —— 一局对局内逐帧记录原始分量，验证 kill/dead/hp 语义。

验证目标：
1. **kill 是否为"我方击杀敌方英雄"**：我方击杀瞬间 kill 分量应跳 -0.5
   （gamecore 已预乘官方权重），且 dead 分量不同步变化。
   若是 => 去加权后 raw=+1/杀，新奖励设计给 +2.0/杀（正奖励）✓
2. **dead 是否为"我方英雄死亡"**：我方死亡瞬间 dead 分量应跳 -1.0，
   且 kill 分量不同步变化。
   若是 => 去加权后 raw=+1/死，新奖励设计给 -1.0/死（负惩罚）✓
3. **kill/dead 事件互斥**：两者都是"我方自身事件计数"，语义独立。
4. 顺带观察 **hp 分量在击杀/死亡瞬间的跳变方向**，判定 hp/tower 分量的
   符号口径（Δ(我方-敌方) / Δ(敌方-我方) / 绝对量），据此恢复 reward.py
   里暂禁用的换血项与推塔项符号（见脚本末尾的判定表）。
5. 对局结束时打印**敌我英雄血量与 req_pb 结构**，验证胜负判定：
   若被记为 win 的对局敌我英雄都活着（且 req_pb 里能翻到塔血量字段），
   说明结束条件是推塔，_terminal_outcome 的判定有误，需要改用塔血判定。

⚠️ 运行前请确认没有训练任务占用 gamecore（runtime_id 相同会冲突）::

    conda activate hok
    python scripts/probe_reward.py --config configs/default.yaml --steps 6000
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hok1v1_rl.config import load_config
from hok1v1_rl.action_mask import split_legal_action_expanded

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("probe_reward")

#: 9 维返回向量中的分量下标（与 reward.py 的 REWARD_COMPONENT_NAMES 一致）
IDX = {"money": 0, "exp": 1, "hp": 2, "ep": 3, "kill": 4, "dead": 5,
       "tower": 6, "last_hit": 7}


def random_legal_action(state):
    """随机取一个完全合法的 6 维动作（与 test_env.py 一致，避免非法动作告警）。"""
    la172 = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)
    splits = split_legal_action_expanded(la172)
    button = int(random.choice(
        [j for j in range(12) if float(splits[0][j]) == 1.0]
    ))
    target_row = splits[5][button]
    legal_targets = [j for j in range(8) if float(target_row[j]) == 1.0]
    return [
        button,
        int(random.choice([j for j in range(16) if float(splits[1][j]) == 1.0])),
        int(random.choice([j for j in range(16) if float(splits[2][j]) == 1.0])),
        int(random.choice([j for j in range(16) if float(splits[3][j]) == 1.0])),
        int(random.choice([j for j in range(16) if float(splits[4][j]) == 1.0])),
        int(random.choice(legal_targets)) if legal_targets else 0,
    ]


def _dump_attrs(obj, prefix):
    """打印对象的属性名与简单值（跳过下划线私有属性），用于发现字段名。"""
    try:
        for a in sorted(x for x in dir(obj) if not x.startswith("_")):
            try:
                v = getattr(obj, a)
            except Exception:      # noqa: BLE001
                continue
            if callable(v):
                continue
            logger.info("    %s.%s = %s", prefix, a, repr(v)[:120])
    except Exception as exc:      # noqa: BLE001
        logger.warning("%s 属性解析失败：%s", prefix, exc)


def print_gameover_heroes(states):
    """对局结束时打印敌我英雄/塔的完整属性（验证胜负判定 + 找塔血字段）。

    若双方英雄都活着（且塔血量可读），说明对局由推塔/时间结束，
    _terminal_outcome 的"我方英雄活着=win"是无效判定，需要改用塔血判定。
    """
    try:
        state = states[0]
        req_pb = state.get("req_pb")
        if req_pb is None:
            logger.info("gameover 时拿不到 req_pb，无法核对英雄血量")
            return
        for i, hero in enumerate(req_pb.hero_list):
            logger.info("hero[%d] camp=%s hp=%s", i, hero.camp, hero.hp)
            _dump_attrs(hero, f"hero[{i}]")
        for i, o in enumerate(req_pb.organ_list):
            logger.info("organ[%d]:", i)
            _dump_attrs(o, f"organ[{i}]")
    except Exception as exc:      # noqa: BLE001
        logger.warning("hero 血量解析失败：%s", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=6000,
                        help="单局最大步数（随机动作，凑足击杀/死亡事件即可）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path
    cfg = load_config(str(cfg_path))

    from hok1v1_rl.env_runner import build_hok_env, make_camp_config

    env = build_hok_env(cfg)
    camp_config = make_camp_config()
    use_common_ai = [True] * cfg.env.player_num
    use_common_ai[0] = False
    obs, reward, done, states = env.reset(
        camp_config, use_common_ai=use_common_ai, eval=False
    )
    player_id = 0

    # ---- 校验 raw10 -> ret9 的切片关系（reward.py 的解析前提）----
    raw10 = np.asarray(states[player_id]["reward"], dtype=np.float32)
    ret9 = np.asarray(reward[player_id], dtype=np.float32)
    logger.info("raw 10 维 = %s", raw10)
    logger.info("ret  9 维 = %s", ret9)
    expect = np.concatenate([raw10[:6], raw10[7:]])
    if np.allclose(ret9, expect, atol=1e-6):
        logger.info("切片关系确认：ret9 == raw10[:6] + raw10[7:]")
    else:
        logger.error("切片关系不匹配！期望 %s 实际 %s，解析前提失效", expect, ret9)

    prev = ret9.astype(np.float64).copy()
    kill_events, dead_events = [], []
    steps = 0
    last_t = time.time()
    while steps < args.steps and not done[player_id]:
        action6 = random_legal_action(states[player_id])
        actions = [tuple([0] * 6)] * cfg.env.player_num
        actions[player_id] = tuple(int(x) for x in action6)
        _obs, reward, done, states = env.step(actions)
        cur = np.asarray(reward[player_id], dtype=np.float32)
        cur_raw = np.asarray(states[player_id]["reward"], dtype=np.float32)
        d = cur - prev
        now = time.time()
        dt_ms = int((now - last_t) * 1000)
        last_t = now

        if d[IDX["kill"]] != 0.0 or d[IDX["dead"]] != 0.0:
            # ret9 的差分 + 同一步 state 里的原始 10 维向量（raw10）。
            # 若 ±X 成对是 env 差分造成，raw10 里只会看到事件本身没有反冲；
            # 若 raw10 也成对，说明 gamecore 原生就这样发。
            logger.info(
                "step %5d (+%4dms) | Δkill %+7.2f | Δdead %+7.2f | Δhp %+9.1f |"
                " Δtower %+9.1f | Δmoney %+6.2f Δexp %+6.2f Δep %+6.2f Δlh %+5.2f |"
                " raw10=%s",
                steps, dt_ms, d[IDX["kill"]], d[IDX["dead"]], d[IDX["hp"]],
                d[IDX["tower"]], d[IDX["money"]], d[IDX["exp"]],
                d[IDX["ep"]], d[IDX["last_hit"]], cur_raw,
            )
        if d[IDX["kill"]] != 0.0:
            kill_events.append((steps, float(d[IDX["kill"]]), float(d[IDX["hp"]])))
        if d[IDX["dead"]] != 0.0:
            dead_events.append((steps, float(d[IDX["dead"]]), float(d[IDX["hp"]])))
        prev = cur
        steps += 1

    logger.info("=" * 70)
    logger.info("对局结束：%d 步，done=%s", steps, done[player_id])
    logger.info("结束帧 raw10 = %s", np.asarray(
        states[player_id]["reward"], dtype=np.float32))
    print_gameover_heroes(states)

    # ---- 判定 ----
    logger.info("=" * 70)
    logger.info("kill 事件 %d 次，Δkill 取值: %s",
                len(kill_events), sorted({e[1] for e in kill_events}))
    logger.info("dead 事件 %d 次，Δdead 取值: %s",
                len(dead_events), sorted({e[1] for e in dead_events}))

    same_step = [k for k in kill_events if any(d[0] == k[0] for d in dead_events)]
    logger.info("kill 与 dead 同帧触发 %d 次（应为 0，互斥 => 各自独立计数）",
                len(same_step))

    if kill_events and dead_events:
        # hp 分量口径判定表（击杀 = 敌方死，死亡 = 我方死）：
        #   击杀瞬间跳正 + 死亡瞬间跳负 => Δ(我方-敌方)，新设计用 +0.001
        #   击杀瞬间跳负 + 死亡瞬间跳正 => Δ(敌方-我方)，必须用 -0.001
        #   击杀瞬间 ≈0  + 死亡瞬间跳负 => 我方绝对血量变化，用 +0.001
        #   （tower 分量同口径，+0.01 对应前两种的符号规则）
        hp_kill = [e[2] for e in kill_events]
        hp_dead = [e[2] for e in dead_events]
        logger.info("击杀瞬间 hp 分量跳变: %s", [round(v, 1) for v in hp_kill])
        logger.info("死亡瞬间 hp 分量跳变: %s", [round(v, 1) for v in hp_dead])
        if all(v > 0 for v in hp_kill) and all(v < 0 for v in hp_dead):
            logger.info("hp 口径 = Δ(我方-敌方)：换血项恢复 +0.001，推塔项恢复 +0.01")
        elif all(v < 0 for v in hp_kill) and all(v > 0 for v in hp_dead):
            logger.info("hp 口径 = Δ(敌方-我方)：换血项恢复 -0.001，推塔项恢复 -0.01")
        elif all(abs(v) < 5 for v in hp_kill) and all(v < 0 for v in hp_dead):
            logger.info("hp 口径 = 我方绝对血量变化：换血项恢复 +0.001（退化为只惩罚掉血）")
        else:
            logger.info("hp 跳变方向不一致（%s / %s），把原始输出发给我人工判断",
                        hp_kill, hp_dead)
    elif dead_events:
        logger.info("本局没有击杀事件，无法判定 hp 口径，请多跑几局")

    all_neg_kill = all(e[1] < 0 for e in kill_events)
    all_neg_dead = all(e[1] < 0 for e in dead_events)
    logger.info("-" * 70)
    logger.info("结论：")
    logger.info("  kill 每击杀恒为负（官方预乘 -0.5）: %s -> 新设计给 +2.0/杀（正奖励）",
                "确认" if all_neg_kill else "存疑，请把上面的原始输出发给我")
    logger.info("  dead 每死亡恒为负（官方预乘 -1.0）: %s -> 新设计给 -1.0/死（负惩罚）",
                "确认" if all_neg_dead else "存疑，请把上面的原始输出发给我")
    logger.info("  kill/dead 互斥: %s", "确认" if not same_step else "存疑")
    logger.info("  若本局没凑够事件（随机动作可能一局没击杀），多跑几次或调大 --steps")

    try:
        env.close_game()
    except Exception:      # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
