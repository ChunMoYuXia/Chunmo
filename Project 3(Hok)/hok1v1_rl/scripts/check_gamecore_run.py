"""使用真实 runner 验证 gamecore；输出状态、奖励分量、终止证据。

默认只采样 64 步；--full-game 跑完一局；--evaluation 使用 SDK eval=True。
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from hok1v1_rl.config import load_config
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.matchups import Matchup
from hok1v1_rl.network import HoK1v1Network
from train_ppo import append_json, select_device, set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--decisions", type=int, default=64)
    parser.add_argument("--full-game", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime-id", default="hok1v1rl-check")
    parser.add_argument("--zmq-port", type=int, default=35460)
    parser.add_argument("--output", help="逐局 JSONL 文件；默认创建到 runs 下")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    cfg = load_config(args.config)
    cfg.env.runtime_id, cfg.env.zmq_port_begin = args.runtime_id, args.zmq_port
    cfg.env.max_decisions = 0 if args.full_game else args.decisions
    cfg.validate()
    if args.games <= 0 or args.decisions <= 0:
        raise ValueError("games/decisions 必须为正")
    set_global_seed(cfg.train.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    network = HoK1v1Network(cfg).to(device)
    output = Path(args.output) if args.output else ROOT / "runs" / ("gamecore-check-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    runner = None
    try:
        runner = EnvRunner(build_hok_env(cfg, cfg.env.runtime_id), network, cfg, device)
        for index in range(args.games):
            # 使用不同英雄，并交替控制双方，验证 SDK 两侧的观测/动作路径。
            hero = cfg.env.hero_ids[index % len(cfg.env.hero_ids)]
            opponent = cfg.env.opponent_hero_ids[(index + 1) % len(cfg.env.opponent_hero_ids)]
            match = Matchup(hero, opponent, index % 2)
            trajectory = runner.collect_trajectory(matchup=match, evaluation=args.evaluation,
                                                   deterministic=args.evaluation)
            stats = trajectory["episode_stats"]
            append_json(output, {"game": index + 1, "evaluation": args.evaluation, **stats})
            print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    finally:
        if runner:
            runner.close()
    print(f"真实对局验证已保存：{output}")

if __name__ == "__main__":
    main()
