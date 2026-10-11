"""训练入口：采样 → PPO → 定期评估 → 保存。请从项目根目录运行。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from hok1v1_rl.config import Config, load_config, ENVIRONMENT_CONTRACT
from hok1v1_rl.env_runner import EnvRunner, build_hok_env
from hok1v1_rl.evaluation import evaluate
from hok1v1_rl.learner import PPOLearner, read_checkpoint
from hok1v1_rl.live_status import LiveStatus
from hok1v1_rl.logger import UnifiedLogger
from hok1v1_rl.metrics_store import MetricsStore
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.policy_diagnostics import merge_policy_diagnostics
from hok1v1_rl.behavior_diagnostics import BEHAVIOR_KEYS
from hok1v1_rl.policy_analysis import action_advantages
from hok1v1_rl.reward_diagnostics import summarize_reward_detection
from hok1v1_rl.rollout import collect_batch
from hok1v1_rl.parallel_sampler import ParallelSampler
from hok1v1_rl.vram_guard import VRamGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("train")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        # 真正执行一次 kernel，ABI/显卡架构问题在启动时暴露。
        matrix = torch.ones(8, 8, device=device)
        (matrix @ matrix).sum().item()
    return device


def append_json(path, record):
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_manifest(directory, cfg, args, initialization=None):
    """冻结本次配置、代码指纹和依赖信息，后续读日志时知道实际训练了什么。"""
    with open(directory / "config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg.to_dict(), stream, allow_unicode=True, sort_keys=False)
    sources = list((ROOT / "hok1v1_rl").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + [ROOT / "config.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    data = {"created_at": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "args": vars(args),
            "config": cfg.to_dict(), "environment_contract": ENVIRONMENT_CONTRACT,
            "simulator_mode": "remote", "initialization": initialization or {"mode": "fresh"},
            "source_sha256": hashes}
    (directory / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def training_metrics(episodes, result, env_steps, elapsed, samples=None):
    # 对局指标只看真正结束的整局；吞吐、动作与耗时只看本批新采到的片段。
    samples = episodes if samples is None else samples
    decisions = sum(e["episode_len"] for e in samples)
    n = len(episodes)
    metrics = {f"train/{key}": float(value) for key, value in asdict(result).items()}
    metrics.update({"env/completed_games": n, "env/frames_per_update": decisions,
                    "env/decisions_per_update": decisions, "env/total_env_steps": env_steps,
                    "env/chunk_padded_steps": sum(e["chunk_padded_steps"] for e in samples),
                    "sample/reward_per_decision": sum(e["reward"] for e in samples) / decisions,
                    "res/fps": decisions / max(elapsed, 1e-8)})
    metrics.update({f"detect/{key}": value for key, value in summarize_reward_detection(samples).items()
                    if value is not None})
    if n:
        metrics.update({"env/win_rate": sum(e["win"] for e in episodes) / n,
                        "env/avg_episode_len": sum(e["episode_len"] for e in episodes) / n,
                        "env/avg_reward": sum(e["reward"] for e in episodes) / n,
                        "env/truncated_rate": sum(e["truncated"] for e in episodes) / n,
                        "env/unknown_rate": sum(e["outcome"] == "unknown" for e in episodes) / n})
        for key in episodes[0]["reward_components"]:
            metrics[f"env/reward/{key}"] = sum(e["reward_components"].get(key, 0) for e in episodes) / n
    for head in range(6):
        counts = np.sum([e["action_hist"][f"head_{head}"] for e in samples], axis=0)
        for choice, count in enumerate(counts):
            metrics[f"act_dist/head_{head}/{choice}"] = float(count / counts.sum())
    diagnostic = merge_policy_diagnostics(samples)
    if diagnostic:
        for key in ("legal_rate", "probability_mean", "selection_rate", "greedy_rate"):
            for button, value in enumerate(diagnostic[key]):
                metrics[f"policy/button_{key}/{button}"] = value
        metrics["policy/near_tie_rate"] = diagnostic["near_tie_rate"]
        metrics["policy/idle_with_alternative_rate"] = diagnostic["idle_with_alternative_rate"]
    for key in ("reset_seconds", "close_game_seconds", "start_game_seconds",
                "inference_seconds", "env_step_seconds"):
        metrics[f"res/{key}"] = sum(e.get("timings", {}).get(key, 0.0) for e in samples)
    counts = {key: sum(e.get("state_diagnostics", {}).get(key, 0) for e in samples)
              for key in ("position_known_decisions", "fountain_decisions", "hero_skill_attempts",
                          "hero_skill_attempts_with_known_position", "hero_skill_attempts_in_fountain",
                          "attack_attempts", "attack_without_enemy_target",
                          "attack_target_constraint_decisions", "attack_button_blocked_decisions", *BEHAVIOR_KEYS)}
    metrics.update({f"behavior/{key}": value for key, value in counts.items()})
    if counts["position_known_decisions"]:
        metrics["behavior/fountain_rate"] = counts["fountain_decisions"] / counts["position_known_decisions"]
    if counts["hero_skill_attempts_with_known_position"]:
        metrics["behavior/fountain_skill_attempt_rate"] = counts["hero_skill_attempts_in_fountain"] / counts["hero_skill_attempts_with_known_position"]
    if counts["attack_attempts"]:
        metrics["behavior/attack_without_enemy_target_rate"] = counts["attack_without_enemy_target"] / counts["attack_attempts"]
    if counts["move_position_known_attempts"]:
        metrics["behavior/move_observed_displacement_rate"] = counts["move_observed_displacements"] / counts["move_position_known_attempts"]
    if counts["observed_frame_span"]:
        for key in ("income_gain", "enemy_income_gain", "hero_damage_gain", "enemy_building_damage"):
            metrics[f"behavior/{key}_per_1000_frames"] = counts[key] * 1000 / counts["observed_frame_span"]
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--total-steps", type=int, default=1000, help="目标累计 PPO 更新次数；续训从已保存次数继续")
    parser.add_argument("--device", default="auto")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="严格恢复配置、Adam、计数和随机状态")
    initialization.add_argument("--init-from", help="只加载旧权重，开启可修改 PPO 或动作约束的新实验")
    parser.add_argument("--allow-reward-change", action="store_true",
                        help="仅搭配 --init-from，显式允许奖励迁移并记录原/新权重")
    parser.add_argument("--runtime-id")
    parser.add_argument("--zmq-port", type=int)
    parser.add_argument("--gamecore-addr")
    parser.add_argument("--log-dir", help="新运行目录；默认自动创建 runs/时间戳")
    parser.add_argument("--max-decisions", type=int, help="仅调试：每局采样上限，不建议长训设置")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--eval-interval", type=int, help="0 表示关闭训练中快评")
    parser.add_argument("--eval-episodes", type=int, help="训练中快评局数；短测可设 2，正式训练默认 40")
    parser.add_argument("--samplers", type=int, default=1, help="独立采样进程数；推荐先设 2，默认 1 保留串行模式")
    parser.add_argument("--sampler-device", default="cpu", help="并行采样的推理设备，默认 cpu；PPO 仍使用 --device")
    parser.add_argument("--sampler-inference", choices=("eager", "jit"), default="eager",
                        help="jit 加速 CPU 单步推理；更换模式时用 --init-from 开启新运行")
    parser.add_argument("--fragment-length", type=int, default=0,
                        help="并行采样每个片段的决策上限；0=整局，推荐片段 512")
    parser.add_argument("--burn-in", type=int, default=64,
                        help="片段模式每次换权重后重放最近多少步来更新 LSTM 记忆")
    return parser.parse_args()


def main():
    args = parse_args()
    allow_reward_change = getattr(args, "allow_reward_change", False)
    if allow_reward_change and not args.init_from:
        raise ValueError("--allow-reward-change 只能搭配 --init-from，不能改变严格续训的奖励")
    if args.resume and not args.config:
        cfg = Config.from_dict(read_checkpoint(args.resume)["config"])
    else:
        cfg = load_config(args.config or str(ROOT / "configs/default.yaml"))
    for option, name in ((args.runtime_id, "runtime_id"), (args.zmq_port, "zmq_port_begin"),
                         (args.gamecore_addr, "gamecore_server_addr"), (args.max_decisions, "max_decisions")):
        if option is not None:
            setattr(cfg.env, name, option)
    for option, name in ((args.batch_size, "batch_size"), (args.minibatch_size, "minibatch_size")):
        if option is not None:
            setattr(cfg.ppo, name, option)
    if args.eval_interval is not None:
        cfg.train.eval_interval_steps = args.eval_interval
    if args.eval_episodes is not None:
        cfg.train.eval_episodes = args.eval_episodes
    cfg.validate()
    samplers = getattr(args, "samplers", 1)
    sampler_device = getattr(args, "sampler_device", "cpu")
    inference_mode = getattr(args, "sampler_inference", "eager")
    fragment_length, burn_in = getattr(args, "fragment_length", 0), getattr(args, "burn_in", 64)
    if fragment_length < 0 or (fragment_length and fragment_length % cfg.network.lstm_time_steps) or burn_in < 0:
        raise ValueError("fragment-length 应为 0 或 LSTM 序列长度的正整数倍，burn-in >= 0")
    if fragment_length and (samplers == 1 or cfg.env.max_decisions):
        raise ValueError("片段采样需要 samplers >= 2，且不设置 max-decisions")
    if not 1 <= samplers <= 8 or cfg.env.zmq_port_begin + 2*samplers - 1 > 65535:
        raise ValueError("samplers 范围为 1..8，每个进程需要两个连续 ZMQ 端口")
    eval_workers = min(samplers, max(2, cfg.train.eval_episodes)) if fragment_length and cfg.train.eval_interval_steps else 0
    if cfg.env.zmq_port_begin + 2*(samplers+eval_workers) - 1 > 65535:
        raise ValueError("片段模式还需要为独立评估进程预留连续 ZMQ 端口")
    if torch.device(sampler_device).type not in ("cpu", "cuda"):
        raise ValueError("sampler-device 只支持 cpu 或 cuda")
    if args.total_steps <= 0:
        raise ValueError("total-steps 必须为正")
    run_base = Path(cfg.train.log_dir)
    if not run_base.is_absolute():
        run_base = ROOT / run_base
    directory = Path(args.log_dir) if args.log_dir else run_base / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # 不把多个训练会话追加到同一目录，避免曲线和配置混在一起。
    directory.mkdir(parents=True, exist_ok=False)
    cfg.train.log_dir, cfg.train.tb_log_dir = str(directory), str(directory / "tb")
    set_global_seed(cfg.train.seed)
    device = select_device(args.device)
    if inference_mode == "jit" and torch.device(sampler_device if samplers > 1 else device).type != "cpu":
        raise ValueError("sampler-inference=jit 仅支持 CPU 采样")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    network = HoK1v1Network(cfg).to(device)
    guard = VRamGuard(cfg.hardware.vram_warn_gb, cfg.hardware.vram_degrade_gb, cfg.hardware.vram_oom_gb)
    learner = PPOLearner(network, cfg, vram_guard=guard)
    state = learner.load(args.resume) if args.resume else {}
    saved_sampler = state.get("sampler_state")
    if saved_sampler and (saved_sampler["workers"] != samplers or saved_sampler["device"] != sampler_device):
        raise ValueError("并行续训需保持 samplers 和 sampler-device 一致；更换时请用 --init-from")
    saved_inference = saved_sampler.get("inference_mode", "eager") if saved_sampler else state.get("sampler_inference", "eager")
    if args.resume and saved_inference != inference_mode:
        raise ValueError("严格续训需保持 sampler-inference 一致；更换时请用 --init-from")
    if args.resume and ((saved_sampler or {}).get("fragment_length", 0) != fragment_length or
            (fragment_length and (saved_sampler or {}).get("burn_in", 64) != burn_in)):
        raise ValueError("严格续训需保持 fragment-length 和 burn-in 一致；更换时请用 --init-from")
    if saved_sampler and saved_sampler.get("ongoing_games"):
        logger.info("恢复模型与优化器；checkpoint 中 %d 局未结束的游戏不恢复，采样将从新局开始",
                    len(saved_sampler["ongoing_games"]))
    initialization = {"mode": "resume", "source_checkpoint": str(Path(args.resume).resolve()),
                      "source_updates": learner._total_updates} if args.resume else {"mode": "fresh"}
    if getattr(args, "init_from", None):
        initialization = learner.initialize_from(args.init_from, allow_reward_changes=allow_reward_change)
    learner.set_lr_schedule(args.total_steps)
    if learner._total_updates >= args.total_steps:
        raise ValueError("目标累计更新数必须大于 checkpoint 的已有更新数")
    write_manifest(directory, cfg, args, initialization)
    tb = UnifiedLogger(cfg.train.tb_log_dir, cfg.train.wandb_project, cfg.train.wandb_enabled, cfg.to_dict())
    store = MetricsStore(str(directory / "metrics.jsonl"))
    runner = None
    env_steps, best_rate = state.get("env_steps", 0), state.get("best_rate", -1.0)
    best_summary = state.get("best_summary")
    status = LiveStatus(directory, phase="starting", target_updates=args.total_steps,
                        completed_updates=learner._total_updates, device=str(device), env_steps=env_steps,
                        sampler_count=samplers)
    status.update(force=True)
    final_phase = "finished"
    logger.info("device=%s 参数=%.2fM run=%s 已完成更新=%d", device, network.count_params()/1e6, directory, learner._total_updates)
    if samplers > 1:
        logger.info("并行采样：%d 进程，推理设备=%s，推理模式=%s，ZMQ=%d..%d；每批同步权重后采样，全部返回后更新",
                    samplers, sampler_device, inference_mode,
                    cfg.env.zmq_port_begin, cfg.env.zmq_port_begin+2*samplers-1)
        if fragment_length:
            logger.info("连续片段：每段最多 %d 决策，burn-in=%d；评估使用独立 runtime 和后续端口",
                        fragment_length, burn_in)
    def checkpoint_state():
        return {"env_steps": env_steps, "best_rate": best_rate, "best_summary": best_summary,
                "sampler_inference": inference_mode,
                "matchup_schedule": runner.schedule.state_dict() if runner else state.get("matchup_schedule"),
                "sampler_state": runner.state_dict() if isinstance(runner, ParallelSampler) else saved_sampler}
    try:
        if samplers > 1:
            runner = ParallelSampler(cfg, network, samplers, sampler_device, saved_sampler,
                                     inference_mode=inference_mode,
                                     fragment_length=fragment_length, burn_in=burn_in)
        else:
            runner = EnvRunner(build_hok_env(cfg, cfg.env.runtime_id), network, cfg, device)
            runner.prepare_inference(inference_mode)
        runner.progress_callback = status.update
        if state.get("matchup_schedule"):
            runner.schedule.load_state_dict(state["matchup_schedule"])
        while learner._total_updates < args.total_steps:
            started = time.monotonic()
            update = learner._total_updates + 1
            status.update(force=True, phase="sampling", update=update, hero_id=None,
                          episode_decisions=0, eval_completed=0)
            def record_episode(game):
                nonlocal env_steps
                if not fragment_length:
                    env_steps += game["episode_len"]
                status.update(force=True, env_steps=env_steps)
                append_json(directory / "episodes.jsonl", {"update": update, "mode": "train", **game})
                logger.info("采样 hero=%s vs=%s side=%s result=%s decisions=%d", game.get("hero_id"), game.get("opponent_id"), game.get("player_id"), game["termination_reason"], game["episode_len"])
            def record_sample(sample):
                nonlocal env_steps
                env_steps += sample["episode_len"]
                status.update(force=True, env_steps=env_steps)
                append_json(directory / "fragments.jsonl", {"update": update, "mode": "train", **sample})
            if samplers > 1:
                batch, episodes = runner.collect_batch(cfg.ppo.batch_size, learner._total_updates,
                    record_episode, record_sample if fragment_length else None)
            else:
                batch, episodes = collect_batch(runner, cfg.ppo.batch_size, record_episode)
            sampling_seconds = time.monotonic() - started
            status.update(force=True, phase="updating")
            updating_started = time.monotonic()
            append_json(directory / "policy_updates.jsonl", {"update": update,
                        "phase": "pre_update", **action_advantages(batch)})
            result = learner.update(batch)
            if samplers == 1 and inference_mode == "jit":
                runner.prepare_inference(inference_mode)
            updating_seconds = time.monotonic() - updating_started
            status.update(force=True, completed_updates=learner._total_updates)
            metrics = training_metrics(episodes, result, env_steps, time.monotonic() - started,
                runner.collection_samples if fragment_length else None)
            metrics.update({"res/sampling_seconds": sampling_seconds, "res/update_seconds": updating_seconds,
                            "res/evaluation_seconds": 0.0, "res/sampler_processes": samplers})
            if isinstance(runner, ParallelSampler):
                metrics.update(runner.collection_metrics)
            if fragment_length:
                metrics.update({"sample/fragments": len(runner.collection_samples),
                                "sample/fragment_length": fragment_length, "sample/burn_in": burn_in})
            if device.type == "cuda":
                metrics.update({"res/vram_used_gb": torch.cuda.memory_allocated(device) / 1024**3,
                                "res/vram_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3})
            if cfg.train.eval_interval_steps and update % cfg.train.eval_interval_steps == 0:
                evaluation_started = time.monotonic()
                eval_count = 0
                status.update(force=True, phase="evaluating", eval_completed=0,
                              eval_target=cfg.train.eval_episodes, hero_id=None, episode_decisions=0)
                def record_eval(game):
                    nonlocal eval_count
                    eval_count += 1
                    status.update(force=True, eval_completed=eval_count)
                    append_json(directory / "episodes.jsonl", {"update": update, "mode": "eval", **game})
                    logger.info("评估 hero=%s side=%s outcome=%s", game["hero_id"], game["player_id"], game["outcome"])
                if samplers > 1:
                    summary = runner.evaluate(cfg.train.eval_episodes, learner._total_updates, record_eval)
                else:
                    summary = evaluate(runner, cfg.train.eval_episodes, on_episode=record_eval)
                metrics["res/evaluation_seconds"] = time.monotonic() - evaluation_started
                append_json(directory / "evaluations.jsonl", {"update": update, **summary})
                metrics.update({"eval/win_rate": summary["win_rate"], "eval/games": summary["games"],
                                "eval/unknown": summary["unknown"], "eval/ci95_lower": summary["win_rate_ci95"][0]})
                diagnostic = summary.get("policy_diagnostics", {})
                if diagnostic:
                    metrics.update({"eval/none2_selection_rate": diagnostic["selection_rate"][1],
                                    "eval/attack_selection_rate": diagnostic["selection_rate"][3],
                                    "eval/attack_legal_rate": diagnostic["legal_rate"][3]})
                if summary["unknown"] == 0 and summary["truncated"] == 0 and summary["win_rate"] > best_rate:
                    best_rate, best_summary = summary["win_rate"], summary
                    learner.save(directory / "best_ckpt.pt", checkpoint_state())
            store.add(update, metrics)
            tb.log_metrics(metrics, update)
            if update % max(1, cfg.train.log_interval_steps) == 0:
                win_text = f"{metrics['env/win_rate']*100:.1f}%" if episodes else "--"
                reward_text = f"{metrics['env/avg_reward']:.2f}" if episodes else "--"
                logger.info("update=%d env_steps=%d completed_games=%d train_win=%s reward=%s KL=%.5f optimizer_steps=%d",
                            update, env_steps, len(episodes), win_text, reward_text, result.kl, result.num_minibatches)
            if update % cfg.train.save_interval_steps == 0:
                status.update(force=True, phase="saving")
                learner.save(directory / f"ckpt_step_{update}.pt", checkpoint_state())
    except KeyboardInterrupt:
        final_phase = "interrupted"
        raise
    except Exception as exc:
        final_phase = "failed"
        status.update(force=True, error=str(exc))
        raise
    finally:
        status.update(force=True, phase="saving")
        try:
            if runner:
                try:
                    runner.close()
                    if isinstance(runner, ParallelSampler):
                        for game in runner.interrupted_episodes:
                            append_json(directory / "interrupted_episodes.jsonl",
                                        {"mode": "interrupted", "update": learner._total_updates, **game})
                finally:
                    learner.save(directory / "final_ckpt.pt", checkpoint_state())
        except BaseException as exc:
            final_phase = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            status.update(force=True, error=str(exc))
            raise
        finally:
            tb.close()
            status.update(force=True, phase=final_phase, completed_updates=learner._total_updates)
    logger.info("训练已保存：%s", directory / "final_ckpt.pt")

if __name__ == "__main__":
    main()
