"""独立评估 checkpoint：SDK eval=True，逐局证据与分组胜率。"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.evaluation import evaluate
from hok1v1_rl.learner import read_checkpoint
from hok1v1_rl.network import HoK1v1Network
from train_ppo import append_json, select_device, set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, help="默认 checkpoint 的 final_episodes（800）")
    parser.add_argument("--seed", type=int, help="阵容种子，默认与训练快评不同")
    parser.add_argument("--action-selection", choices=("argmax", "sample"), default="argmax",
                        help="动作选择方式；默认保持正式评估的 argmax 口径")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime-id", default="hok1v1rl-eval")
    parser.add_argument("--zmq-port", type=int, default=35250)
    parser.add_argument("--gamecore-addr")
    parser.add_argument("--output", help="新建评估目录")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    checkpoint = read_checkpoint(args.checkpoint)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.env.runtime_id, cfg.env.zmq_port_begin = args.runtime_id, args.zmq_port
    cfg.env.max_decisions = 0  # 独立评估始终跑完整游戏
    if args.gamecore_addr:
        cfg.env.gamecore_server_addr = args.gamecore_addr
    cfg.validate()
    episodes = cfg.evaluation.final_episodes if args.episodes is None else args.episodes
    if episodes <= 0:
        raise ValueError("episodes 必须为正")
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / ("eval-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    seed = cfg.evaluation.seed + 1 if args.seed is None else args.seed
    set_global_seed(seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    network = HoK1v1Network(cfg).to(device)
    network.load_state_dict(checkpoint["network"])
    runner, count = None, 0
    def record(game):
        nonlocal count
        count += 1
        append_json(output / "games.jsonl", {"game": count, **game})
        logging.info("%d/%d hero=%d vs=%d side=%d outcome=%s", count, episodes, game["hero_id"], game["opponent_id"], game["player_id"], game["outcome"])
    try:
        runner = EnvRunner(build_hok_env(cfg, cfg.env.runtime_id), network, cfg, device)
        summary = evaluate(runner, episodes, seed, record, action_selection=args.action_selection)
    finally:
        if runner:
            runner.close()
    summary.update(checkpoint=str(Path(args.checkpoint).resolve()), updates=checkpoint["updates"],
                   max_frame_num=cfg.env.max_frame_num, predict_frequency=cfg.env.predict_frequency)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"评估已保存：{output}")

if __name__ == "__main__":
    main()
