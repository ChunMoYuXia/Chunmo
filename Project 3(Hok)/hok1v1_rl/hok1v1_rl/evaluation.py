"""真实评估模式：固定英雄配对、双方阵营、逐局证据、保守胜率分母。"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import math
import random
import numpy as np
import torch
from .matchups import evaluation_schedule
from .policy_diagnostics import merge_policy_diagnostics
from .behavior_diagnostics import summarize_behavior
from .reward_diagnostics import summarize_reward_detection


def wilson_interval(wins, games):
    """95% Wilson 区间；小样本的 9/10 不能证明真实胜率超过 90%。"""
    if games == 0:
        return [0.0, 1.0]
    z, rate = 1.959963984540054, wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_games(games, goal=0.90, required_games=800):
    outcomes = Counter(game["outcome"] for game in games)
    n, wins = len(games), outcomes["win"]
    by_hero, by_side = defaultdict(list), defaultdict(list)
    for game in games:
        by_hero[str(game["hero_id"])].append(game)
        by_side[str(game["player_id"])].append(game)
    def group_summary(group):
        return {key: {"games": len(rows), "wins": sum(r["win"] for r in rows),
                      "lose": sum(r["outcome"] == "lose" for r in rows),
                      "draw": sum(r["outcome"] == "draw" for r in rows),
                      "unknown": sum(r["outcome"] == "unknown" for r in rows),
                      "truncated": sum(r["truncated"] for r in rows),
                      "win_rate": sum(r["win"] for r in rows) / len(rows)}
                for key, rows in sorted(group.items())}
    interval = wilson_interval(wins, n)
    truncated = sum(g["truncated"] for g in games)
    return {"games": n, "wins": wins, "losses": outcomes["lose"],
            "draws": outcomes["draw"], "unknown": outcomes["unknown"],
            "truncated": truncated, "win_rate": wins / n if n else 0.0,
            "win_rate_ci95": interval, "by_hero": group_summary(by_hero),
            "by_side": group_summary(by_side),
            "goal": goal, "required_games": required_games,
            "policy_diagnostics": merge_policy_diagnostics(games),
            "behavior": summarize_behavior(games),
            "reward_detection": summarize_reward_detection(games),
            "goal_verified": n >= required_games and interval[0] >= goal and
                             outcomes["unknown"] == 0 and truncated == 0}


@contextmanager
def preserve_training_rng():
    """SDK 也可能使用全局 RNG；评估后恢复训练随机流。"""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate(runner, episodes, seed=None, on_episode=None, action_selection="argmax"):
    if action_selection not in ("argmax", "sample"):
        raise ValueError("action_selection 必须为 argmax 或 sample")
    cfg = runner.cfg
    seed = cfg.evaluation.seed if seed is None else seed
    schedule = evaluation_schedule(cfg.env.hero_ids, cfg.env.opponent_hero_ids, seed, episodes)
    games = []
    was_training = runner.network.training
    with preserve_training_rng():
        try:
            for match in schedule:
                trajectory = runner.collect_trajectory(deterministic=action_selection == "argmax",
                                                       matchup=match, evaluation=True)
                game = trajectory["episode_stats"]
                games.append(game)
                if on_episode:
                    on_episode(game)
        finally:
            runner.close()
            runner.network.train(was_training)
    summary = summarize_games(games, cfg.evaluation.goal_win_rate, cfg.evaluation.final_episodes)
    summary.update(seed=seed, opponent="common_ai", action_selection=action_selection,
                   hero_ids=cfg.env.hero_ids, opponent_hero_ids=cfg.env.opponent_hero_ids,
                   action=cfg.to_dict()["action"])
    return summary
