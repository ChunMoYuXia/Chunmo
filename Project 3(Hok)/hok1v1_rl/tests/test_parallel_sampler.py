"""实际 spawn 子进程，验证策略同步、完整轨迹、评估隔离及随机流恢复。"""
import copy
from datetime import datetime
import pickle
import time
import unittest

import torch

from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.config import Config
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.parallel_sampler import ParallelSampler
from test_training import FakeGame


class SlowFakeGame(FakeGame):
    def step(self, actions):
        time.sleep(.05)  # 保证两局确实存在重叠运行的时间。
        return super().step(actions)


def fake_runner(cfg, network, device):
    env = SlowFakeGame(end=4)
    env.runtime_id = cfg.env.runtime_id
    return EnvRunner(env, network, cfg, device)


def failing_runner(cfg, network, device):
    raise RuntimeError("模拟采样进程启动失败")


class ParallelSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        cls.cfg = Config()
        cls.cfg.env.hero_ids, cls.cfg.env.opponent_hero_ids = [112,169], [132,133]
        cls.cfg.network.lstm_hidden = 16
        cls.cfg.network.lstm_time_steps = 4
        cls.cfg.ppo.minibatch_size = 4
        from hok1v1_rl.network import HoK1v1Network
        cls.network = HoK1v1Network(cls.cfg)

    def pool(self, network=None, saved_state=None, inference_mode="eager"):
        return ParallelSampler(copy.deepcopy(self.cfg), network or copy.deepcopy(self.network),
                               workers=2, saved_state=saved_state, runner_factory=fake_runner,
                               timeout=30, inference_mode=inference_mode)

    def test_concurrent_games_policy_sync_and_no_padding_count(self):
        network = copy.deepcopy(self.network)
        pool = self.pool(network)
        try:
            batch, games = pool.collect_batch(1, version=0)
            self.assertEqual(len(games), 2)
            self.assertEqual({g["worker_id"] for g in games}, {0, 1})
            self.assertEqual({g["policy_version"] for g in games}, {0})
            self.assertEqual(int(batch["valid_mask"].sum()), sum(g["episode_len"] for g in games))
            self.assertEqual(len({w["process"].pid for w in pool.workers}), 2)
            self.assertEqual([w["cfg"].env.zmq_port_begin for w in pool.workers], [35150, 35152])
            self.assertEqual(len({w["cfg"].env.runtime_id for w in pool.workers}), 2)
            begin = [datetime.fromisoformat(g["started_at"]) for g in games]
            end = [datetime.fromisoformat(g["ended_at"]) for g in games]
            self.assertLess(max(begin), min(end))
            metrics = pool.collection_metrics
            self.assertGreater(metrics["res/weight_sync_seconds"], 0)
            self.assertGreaterEqual(metrics["res/batch_drain_seconds"], 0)
            self.assertTrue(0 <= metrics["res/sampler_idle_fraction"] <= 1)
            self.assertEqual(metrics["res/batch_overshoot_ratio"], 8)
            learner = PPOLearner(network, self.cfg)
            learner.update(batch)  # 更新前 logp 审计会拒绝来自错误策略的轨迹。
            fresh, games = pool.collect_batch(1, version=1)
            self.assertEqual({g["policy_version"] for g in games}, {1})
            self.assertLess(learner._diagnose(fresh, 2)["old_new_logp_max"], 1e-5)
        finally:
            pool.close()
        self.assertTrue(all(not w["process"].is_alive() for w in pool.workers))

    def test_evaluation_preserves_training_rng_and_resume_replays_next_batch(self):
        network = copy.deepcopy(self.network)
        pool = self.pool(network)
        try:
            pool.collect_batch(1, version=0)
            saved = copy.deepcopy(pool.state_dict())
            schedule = copy.deepcopy(pool.schedule.state_dict())
            summary = pool.evaluate(4, version=0)
            self.assertEqual(summary["games"], 4)
            self.assertEqual(summary["action_selection"], "argmax")
            self.assertEqual(pickle.dumps(saved), pickle.dumps(pool.state_dict()))
            self.assertEqual(schedule, pool.schedule.state_dict())
            expected, expected_games = pool.collect_batch(1, version=0)
        finally:
            pool.close()
        restored = self.pool(network, saved)
        try:
            restored.schedule.load_state_dict(schedule)
            actual, games = restored.collect_batch(1, version=0)
            torch.testing.assert_close(actual["actions"], expected["actions"])
            torch.testing.assert_close(actual["log_probs"], expected["log_probs"])
            self.assertEqual([(g["hero_id"],g["opponent_id"],g["player_id"]) for g in games],
                             [(g["hero_id"],g["opponent_id"],g["player_id"]) for g in expected_games])
        finally:
            restored.close()

    def test_worker_startup_error_is_reported_to_parent(self):
        with self.assertRaisesRegex(RuntimeError, "模拟采样进程启动失败"):
            ParallelSampler(copy.deepcopy(self.cfg), copy.deepcopy(self.network), workers=2,
                            runner_factory=failing_runner, timeout=30)

    def test_jit_workers_refresh_weights_after_update(self):
        network = copy.deepcopy(self.network)
        pool = self.pool(network, inference_mode="jit")
        try:
            learner = PPOLearner(network, self.cfg)
            for version in (0, 1):
                batch, games = pool.collect_batch(1, version=version)
                self.assertEqual({g["sampler_inference"] for g in games}, {"jit"})
                self.assertLess(learner._diagnose(batch, 2)["old_new_logp_max"], 1e-5)
                learner.update(batch)
            self.assertEqual(pool.state_dict()["inference_mode"], "jit")
        finally:
            pool.close()

    def test_invalid_worker_count_ports_and_resume_topology(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.env.zmq_port_begin = 65534
        with self.assertRaises(ValueError):
            ParallelSampler(cfg, self.network, workers=2)
        with self.assertRaisesRegex(ValueError, "samplers"):
            ParallelSampler(self.cfg, self.network, workers=2,
                            saved_state={"workers": 3, "device": "cpu", "rng": []})
        with self.assertRaisesRegex(ValueError, "sampler-inference"):
            self.pool(saved_state={"workers": 2, "device": "cpu", "rng": []}, inference_mode="jit")
