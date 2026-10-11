"""短局诊断：比较采样/argmax 的合法动作与概率，测量策略/价值梯度。

默认截断对局只用于诊断，不报告达标胜率。原 checkpoint 始终只读。
"""
import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.learner import PPOLearner, read_checkpoint, normalize_advantages
from hok1v1_rl.matchups import Matchup
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.rollout import action_head_weights
from hok1v1_rl.policy_diagnostics import merge_policy_diagnostics
from hok1v1_rl.rollout import join_trajectories
from hok1v1_rl.action_mask import effective_legal_action
from hok1v1_rl.reward import extract_hero_snapshot, organ_snapshot
from hok1v1_rl.behavior_diagnostics import summarize_behavior
from hok1v1_rl.policy_analysis import action_advantages, head_exploration
from hok1v1_rl.reward_diagnostics import summarize_reward_detection
from train_ppo import append_json, select_device, set_global_seed, write_manifest


def command_snapshot(command):
    """记录引擎报告的实际命令；不把策略按钮当成成功施法的证明。"""
    if command is None:
        return None
    if not hasattr(command, "command_type"):
        return [command_snapshot(item) for item in command]
    kind = command.command_type
    name = getattr(kind, "name", str(kind))
    result = {"type": int(kind), "name": name}
    fields = {"COMMAND_TYPE_MoveDir": "move_dir", "COMMAND_TYPE_MovePos": "move_pos",
              "COMMAND_TYPE_DirSkill": "dir_skill", "COMMAND_TYPE_PosSkill": "pos_skill",
              "COMMAND_TYPE_ObjSkill": "obj_skill", "COMMAND_TYPE_AttackActor": "attack_actor",
              "COMMAND_TYPE_AttackCommon": "attack_common"}
    detail = getattr(command, fields[name], None) if name in fields else None
    if detail is not None:
        for key in ("degree", "skillID", "slotType", "actorID"):
            if hasattr(detail, key):
                result[key] = int(getattr(detail, key))
    return result


def frame_snapshot(state):
    req = state["req_pb"]
    observer_camp = next(int(h.camp) for h in req.hero_list if h.runtime_id == state["player_id"])
    heroes = []
    for hero in req.hero_list:
        skills = hero.skill
        entries = skills.items() if hasattr(skills, "items") else enumerate(skills)
        heroes.append({"runtime_id": int(hero.runtime_id), "hero_id": int(hero.config_id),
                       "camp": int(hero.camp), "hp": float(hero.hp), "ep": float(hero.ep),
                       "x": float(hero.location.x), "z": float(hero.location.z),
                       "damage": float(hero.totalHurtToHero),
                       "economy_combat": extract_hero_snapshot(req, int(hero.camp), observer_camp),
                       "buildings": organ_snapshot(req, int(hero.camp)),
                       "real_command": command_snapshot(hero.real_cmd),
                       "skills": {str(slot): {"id": int(skill.config_id),
                                              "cooldown": float(skill.cooldown),
                                              "level": int(skill.level)} for slot, skill in entries}})
    mine = next(hero for hero in heroes if hero["runtime_id"] == int(state["player_id"]))
    # 保留原始单位 ID，供实际命令 actorID 核验；不把越界或不可见目标猜成有效目标。
    units = [{"runtime_id": int(unit.runtime_id), "camp": int(unit.camp),
              "hp": float(unit.hp)} for unit in getattr(req, "soldier_list", [])]
    towers = []
    for organ in getattr(req, "organ_list", []):
        if int(organ.type) != 21:
            continue
        location = getattr(organ, "location", None)
        towers.append({"camp": int(organ.camp), "hp": float(organ.hp),
                       "x": float(location.x) if location is not None and hasattr(location, "x") else None,
                       "z": float(location.z) if location is not None and hasattr(location, "z") else None})
    return {"frame": int(req.frame_no), "player_id": mine["runtime_id"],
            "heroes": heroes, "soldiers": units, "towers": towers,
            "sdk_reward_raw": [float(x) for x in state.get("reward", [])]}


class TraceRunner(EnvRunner):
    """只用于诊断：同时保存发送前动作和引擎返回状态，不改变训练采样器。"""
    trace_path = None
    action_selection = None
    trace_game = 0

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.trace_state = state
        self.trace_game += 1
        return state

    def step(self, action):
        if self.trace_path is None:
            return super().step(action)
        before = frame_snapshot(self.trace_state)
        raw = list(map(float, self.trace_state["legal_action"]))
        legal = effective_legal_action(self.trace_state, self.cfg.action)
        weights = action_head_weights(self.trace_state, int(action[0])).tolist()
        result = super().step(action)
        after = frame_snapshot(result[0])
        append_json(self.trace_path, {"game": self.trace_game, "mode": self.action_selection,
                    "sdk_client_id": self.episode_info.get("sdk_client_id"),
                    "action": [int(x) for x in action], "head_weights": weights,
                    "head_probabilities": self.last_head_probabilities,
                    "action_rule": self.cfg.action.rule, "raw_legal_action": raw,
                    "effective_legal_action": legal.tolist(),
                    "state_reward_components": result[3],
                    "before": before, "after": after})
        self.trace_state = result[0]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reward-config", help="只读诊断时从 YAML 覆盖奖励；不更新 checkpoint")
    parser.add_argument("--games", type=int, default=2, help="每种动作模式的对局数，交替双方")
    parser.add_argument("--decisions", type=int, default=512, help="每局诊断上限；0 表示完整比赛")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime-id", default="hok-policy-diagnosis")
    parser.add_argument("--zmq-port", type=int, default=35480)
    parser.add_argument("--output", help="新建诊断目录")
    parser.add_argument("--trace-actions", action="store_true", help="另存逐步坐标、技能冷却、各动作头合法概率和实际命令，仅用于诊断")
    parser.add_argument("--action-rule", choices=("sdk", "targeted_attack_v1"), help="只读比较动作规则；不更新 checkpoint")
    parser.add_argument("--tower-guard-radius", type=float,
                        help="只读测试塔前移动保护半径；0 关闭，不更新 checkpoint")
    parser.add_argument("--hero-ids", nargs="+", type=int, help="诊断英雄子集；不改变正式训练英雄池")
    parser.add_argument("--opponent-ids", nargs="+", type=int,
                        help="指定诊断对手，按每两局一个对手轮换；不改变正式训练对手池")
    parser.add_argument("--gamecore-addr", help="覆盖本机 gamecore 地址")
    parser.add_argument("--modes", nargs="+", choices=("sample", "argmax"), default=["sample", "argmax"],
                        help="可只运行 sample；梯度和优势诊断必须包含随机采样模式")
    args = parser.parse_args()
    if args.games <= 0 or args.decisions < 0:
        parser.error("games 必须为正，decisions 必须非负")
    if "sample" not in args.modes or len(args.modes) != len(set(args.modes)):
        parser.error("modes 必须包含 sample 且不重复；argmax 独立评估请使用 evaluate.py")
    checkpoint = read_checkpoint(args.checkpoint)
    cfg = Config.from_dict(checkpoint["config"])
    if args.reward_config:
        cfg.reward = Config.from_yaml(args.reward_config).reward
    cfg.env.runtime_id, cfg.env.zmq_port_begin = args.runtime_id, args.zmq_port
    cfg.env.max_decisions = args.decisions
    if args.action_rule:
        cfg.action.rule = args.action_rule
    if args.tower_guard_radius is not None:
        cfg.action.tower_guard_radius = args.tower_guard_radius
    if args.gamecore_addr:
        cfg.env.gamecore_server_addr = args.gamecore_addr
    cfg.validate()
    heroes = cfg.env.hero_ids if args.hero_ids is None else args.hero_ids
    if not heroes or not set(heroes) <= set(cfg.env.hero_ids):
        parser.error("诊断英雄必须属于 checkpoint 英雄池")
    opponents = cfg.env.opponent_hero_ids if args.opponent_ids is None else args.opponent_ids
    if not opponents or not set(opponents) <= set(cfg.env.opponent_hero_ids):
        parser.error("诊断对手必须属于 checkpoint 对手英雄池")
    output = Path(args.output) if args.output else ROOT / "runs" / ("policy-diagnosis-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output, cfg, args, {"mode": "diagnostic_only",
                   "source_checkpoint": str(Path(args.checkpoint).resolve()),
                   "source_updates": checkpoint["updates"]})
    set_global_seed(cfg.train.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    network = HoK1v1Network(cfg).to(device)
    network.load_state_dict(checkpoint["network"])
    runner, sampled, groups, behavior, reward_detection = None, [], {}, {}, {}
    try:
        runner = TraceRunner(build_hok_env(cfg, cfg.env.runtime_id), network, cfg, device)
        if args.trace_actions:
            runner.trace_path = output / "action_trace.jsonl"
            runner.capture_action_probabilities = True
        for mode in args.modes:
            runner.action_selection = mode
            games = []
            # 两种模式使用相同阵容和全局种子；gamecore 内部随机性仍未控制。
            set_global_seed(cfg.train.seed)
            for index in range(args.games):
                # 同一英雄先红后蓝；40 局覆盖默认 20 英雄的双方。
                hero_index = index // 2
                opponent_index = (hero_index + 1) % len(opponents) if args.opponent_ids is None else hero_index % len(opponents)
                match = Matchup(heroes[hero_index % len(heroes)],
                                opponents[opponent_index], index % 2)
                trajectory = runner.collect_trajectory(deterministic=mode == "argmax", matchup=match,
                                                       evaluation=True)
                game = trajectory["episode_stats"]
                games.append(game)
                if mode == "sample":
                    sampled.append(trajectory)
                append_json(output / "games.jsonl", {"action_selection": mode, **game})
                print(f"{mode}: hero={match.hero_id} side={match.player_id} decisions={game['episode_len']}", flush=True)
            groups[mode] = merge_policy_diagnostics(games)
            behavior[mode] = summarize_behavior(games)
            reward_detection[mode] = summarize_reward_detection(games)
        batch = normalize_advantages(join_trajectories(sampled))
        # 留下 CPU 张量供离线对照，避免每次检查都重新启动 gamecore。
        torch.save(batch, output / "batch.pt")
        # 诊断阶段只测梯度，既不 step 也不改动模型；使用原 checkpoint 的 PPO 权重。
        learner = PPOLearner(network, cfg)
        learner._total_updates = checkpoint["updates"]
        learner.set_lr_schedule(checkpoint["schedule_total"])
        size = max(1, cfg.ppo.minibatch_size // cfg.network.lstm_time_steps)
        audit = learner._diagnose(batch, size)
        if audit["old_new_logp_max"] > 1e-3:
            raise RuntimeError("诊断 batch 的采样/训练 logp 不一致")
        if not torch.isfinite(torch.tensor(audit["old_new_value_scaled_max"])) or audit["old_new_value_scaled_max"] > 1:
            raise RuntimeError("诊断 batch 的采样/训练 value 不一致")
        gradients = learner.diagnose_gradients(batch, torch.arange(min(size, len(batch["features"]))),
                                               separate_entropy=True)
        result = {"checkpoint": str(Path(args.checkpoint).resolve()), "updates": checkpoint["updates"],
                  "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  "action": cfg.to_dict()["action"], "hero_ids": heroes, "config": cfg.to_dict(),
                  "diagnostic_only": True, "max_decisions": args.decisions,
                  "policy": groups, "behavior": behavior, "gradients": gradients, "probability_audit": audit,
                  "reward_detection": reward_detection,
                  "action_advantages": action_advantages(batch),
                  "head_exploration": head_exploration(learner, batch)}
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        if runner:
            runner.close()
    print(f"诊断已保存：{output}")


if __name__ == "__main__":
    main()
