"""片段边界不是终局：覆盖续接、bootstrap、换权重、评估隔离与中断。"""
import copy
import pickle
import time
import unittest

import numpy as np
import torch

from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.matchups import Matchup
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.parallel_sampler import ParallelSampler
from hok1v1_rl.rollout import EpisodeCollector, join_trajectories
from scripts.train_ppo import training_metrics
from test_training import FakeGame


def fragment_runner(cfg, network, device):
    env = FakeGame(end=6)
    env.runtime_id = cfg.env.runtime_id
    return EnvRunner(env, network, cfg, device)


class UnevenGame(FakeGame):
    def step(self, actions):
        if self.end == 30:
            time.sleep(.005)
        return super().step(actions)


def uneven_runner(cfg, network, device):
    env = UnevenGame(end=5 if cfg.env.runtime_id.endswith("-p0") else 30)
    env.runtime_id = cfg.env.runtime_id
    return EnvRunner(env, network, cfg, device)


class FragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cfg = Config()
        cfg.network.lstm_hidden, cfg.network.lstm_time_steps = 16, 4
        cfg.ppo.minibatch_size = 4
        cfg.env.hero_ids, cfg.env.opponent_hero_ids = [112, 169], [132, 133]
        cls.cfg, cls.network = cfg, HoK1v1Network(cfg).eval()

    def runner(self, end=11):
        return EnvRunner(FakeGame(end=end), copy.deepcopy(self.network),
                         copy.deepcopy(self.cfg), torch.device("cpu"))

    def pool(self, runner_factory=fragment_runner):
        return ParallelSampler(copy.deepcopy(self.cfg), copy.deepcopy(self.network),
                               workers=2, inference_mode="jit", fragment_length=4,
                               burn_in=8, runner_factory=runner_factory, timeout=30)

    def test_splitting_preserves_actions_memory_and_terminal_reward(self):
        whole, split = self.runner(), self.runner()
        matchup = Matchup(112, 132, 0)
        torch.manual_seed(45)
        expected = whole.collect_trajectory(matchup=matchup)
        torch.manual_seed(45)
        collector = EpisodeCollector(split, matchup, burn_in=8)
        fragments = [collector.collect(4, version=0) for _ in range(3)]
        for key in ("features", "actions", "log_probs", "values", "rewards", "dones"):
            actual = np.concatenate([f[key][f["valid_mask"].astype(bool)] for f in fragments])
            np.testing.assert_allclose(actual, expected[key][expected["valid_mask"].astype(bool)], atol=1e-5)
        self.assertEqual(len(split.env.resets), 1)
        self.assertIsNone(fragments[0]["episode_stats"])
        self.assertIsNone(fragments[1]["episode_stats"])
        self.assertFalse(fragments[0]["sample_stats"]["truncated"])
        self.assertFalse(fragments[0]["dones"].any())
        self.assertEqual([f["sample_stats"]["reward_components"]["terminal"] for f in fragments], [0, 0, 50])
        for fragment in fragments[:2]:
            valid = fragment["valid_mask"].astype(bool)
            last_return = fragment["returns"][valid][-1]
            target = fragment["rewards"][valid][-1] + self.cfg.ppo.gamma*fragment["sample_stats"]["bootstrap_value"]
            self.assertAlmostEqual(float(last_return), float(target), places=5)
        self.assertEqual(float(fragments[-1]["dones"].sum()), 1)
        game = fragments[-1]["episode_stats"]
        self.assertEqual(game["episode_len"], 11)
        self.assertEqual(game["fragment_count"], 3)
        self.assertEqual(game["chunk_padded_steps"], 1)
        self.assertAlmostEqual(game["reward"], expected["episode_stats"]["reward"])
        self.assertEqual(game["action_hist"], expected["episode_stats"]["action_hist"])
        new_game = EpisodeCollector(split, Matchup(169, 133, 1), burn_in=8)
        self.assertTrue(all(torch.count_nonzero(s) == 0 for s in new_game.hidden))
        self.assertEqual(len(split.env.resets), 2)

    def test_burn_in_uses_new_weights_without_consuming_rng_and_metrics_use_fragments(self):
        runner = self.runner()
        runner.prepare_inference("jit")
        collector = EpisodeCollector(runner, Matchup(112, 132, 0), burn_in=8)
        first = collector.collect(4, version=0)
        learner = PPOLearner(runner.network, runner.cfg)
        result = learner.update(join_trajectories([first]))
        metrics = training_metrics([], result, 4, 1, [first["sample_stats"]])
        self.assertEqual(metrics["env/decisions_per_update"], 4)
        self.assertEqual(metrics["env/completed_games"], 0)
        self.assertNotIn("env/win_rate", metrics)
        self.assertNotIn("env/avg_reward", metrics)
        runner.prepare_inference("jit")
        rng = torch.get_rng_state().clone()
        collector.refresh_memory()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad():
            features = torch.from_numpy(first["features"])
            expected_hidden = runner.network(features)["lstm_state_new"]
        for actual, expected in zip(collector.hidden, expected_hidden):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        second = collector.collect(4, version=1)
        batch = join_trajectories([second])
        self.assertLess(learner._diagnose(batch, 2)["old_new_logp_max"], 1e-5)
        learner.update(batch)
        last = collector.collect(4, version=1)
        self.assertEqual(last["episode_stats"]["policy_versions"], [0, 1])
        self.assertEqual(len(runner.env.resets), 1)

    def test_processes_continue_same_games_across_updates_and_independent_evaluation(self):
        pool = self.pool()
        try:
            first, games = pool.collect_batch(1, version=0)
            self.assertEqual(games, [])
            self.assertEqual(int(first["valid_mask"].sum()), 8)
            saved = copy.deepcopy(pool.state_dict())
            saved_schedule = copy.deepcopy(pool.schedule.state_dict())
            self.assertEqual(len(saved["ongoing_games"]), 2)
            summary = pool.evaluate(2, version=0)
            self.assertEqual(summary["games"], 2)
            self.assertEqual(pickle.dumps(saved), pickle.dumps(pool.state_dict()))
            self.assertNotEqual(pool.evaluation_pool.workers[0]["cfg"].env.runtime_id,
                                pool.workers[0]["cfg"].env.runtime_id)
            self.assertEqual(pool.evaluation_pool.cfg.env.zmq_port_begin, pool.cfg.env.zmq_port_begin+4)
            observed = []
            second, games = pool.collect_batch(1, version=1, on_episode=observed.append)
            self.assertEqual(len(games), 2)
            self.assertEqual(len(observed), 2)
            self.assertEqual(int(second["valid_mask"].sum()), 4)
            self.assertTrue(all(g["episode_len"] == 6 and g["fragment_count"] == 2 for g in games))
            self.assertEqual({g["sdk_game_id"] for g in games}, {"fake-game-1"})
            self.assertTrue(all(g["policy_versions"] == [0, 1] for g in games))
            third, games = pool.collect_batch(1, version=1)
            self.assertEqual(games, [])
            self.assertTrue(all(torch.count_nonzero(s) == 0 for s in third["lstm_state_init"]))
        finally:
            pool.close()
        self.assertEqual(len(pool.interrupted_episodes), 2)
        self.assertTrue(all(g["outcome"] == "interrupted" and not g["win"] and
                            g["reward_components"]["terminal"] == 0 for g in pool.interrupted_episodes))
        self.assertTrue(all(not w["process"].is_alive() for w in pool.workers+pool.evaluation_pool.workers))
        restored = ParallelSampler(copy.deepcopy(self.cfg), copy.deepcopy(self.network), workers=2,
            saved_state=saved, inference_mode="jit", fragment_length=4, burn_in=8,
            runner_factory=fragment_runner, timeout=30)
        try:
            restored.schedule.load_state_dict(saved_schedule)
            batch, games = restored.collect_batch(1, version=1)
            self.assertEqual(games, [])
            # 恢复 RNG 和阵容安排，但不把旧局的 feature/hidden 安到新局上。
            self.assertEqual(float(batch["features"][0, 0, 0]), 0.0)
            self.assertTrue(all(torch.count_nonzero(s) == 0 for s in batch["lstm_state_init"]))
        finally:
            restored.close()

    def test_configuration_changes_require_new_run(self):
        with self.assertRaisesRegex(ValueError, "fragment-length"):
            ParallelSampler(self.cfg, self.network, workers=2, fragment_length=3)
        with self.assertRaisesRegex(ValueError, "fragment-length"):
            ParallelSampler(self.cfg, self.network, workers=2, fragment_length=4,
                saved_state={"workers": 2, "device": "cpu", "inference_mode": "eager"})

    def test_finished_worker_waits_for_next_batch_instead_of_resetting_in_tail(self):
        pool = self.pool(uneven_runner)
        try:
            batch, games = pool.collect_batch(24, version=0)
            self.assertGreaterEqual(int(batch["valid_mask"].sum()), 24)
            fast_samples = [s for s in pool.collection_samples if s["worker_id"] == 0]
            self.assertEqual({s["sdk_game_id"] for s in fast_samples}, {"fake-game-1"})
            self.assertEqual(sum(s["episode_len"] for s in fast_samples), 5)
            self.assertEqual(len(games), 1)
            # 极短游戏仍可开启新一轮补足 batch，不会因所有进程空闲而死锁。
        finally:
            pool.close()
        pool = self.pool()
        try:
            batch, games = pool.collect_batch(20, version=0)
            self.assertGreaterEqual(int(batch["valid_mask"].sum()), 20)
            self.assertGreaterEqual(len(games), 2)
            self.assertTrue(any(s["sdk_game_id"] == "fake-game-2" for s in pool.collection_samples))
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
