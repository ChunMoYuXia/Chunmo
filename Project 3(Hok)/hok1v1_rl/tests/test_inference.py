"""加速推理必须保持合法动作、概率和记忆状态，并及时使用更新后的权重。"""
import copy
import unittest

import numpy as np
import torch

from hok1v1_rl.action_mask import assert_action_legal
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.rollout import collect_batch
from test_training import FakeGame


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cfg = Config()
        cfg.network.lstm_hidden = 32
        cfg.network.lstm_time_steps = 4
        cfg.ppo.minibatch_size = 4
        cls.cfg = cfg
        cls.network = HoK1v1Network(cfg).eval()

    def runner(self):
        return EnvRunner(FakeGame(end=7), copy.deepcopy(self.network),
                         copy.deepcopy(self.cfg), torch.device("cpu"))

    def test_recurrent_actions_probabilities_and_rng_match_eager(self):
        eager, fast = self.runner(), self.runner()
        rng = torch.get_rng_state().clone()
        fast.prepare_inference("jit")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        hidden_eager = eager.network.init_hidden(1, "cpu")
        hidden_fast = fast.network.init_hidden(1, "cpu")
        for step in range(24):
            feature = np.random.default_rng(step).normal(size=725).astype(np.float32)
            legal = np.ones(172, np.float32)
            legal[:12:2] = 0
            # 不同 button 对应不同的唯一合法 target，检查条件目标选择。
            targets = legal[76:].reshape(12, 8)
            targets[:] = 0
            targets[np.arange(12), np.arange(12) % 8] = 1
            rng = torch.get_rng_state().clone()
            action, logp, hidden_eager, value = eager.sample_action(feature, legal, hidden_eager)
            expected_rng = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            other, other_logp, hidden_fast, other_value = fast.sample_action(feature, legal, hidden_fast)
            np.testing.assert_array_equal(other, action)
            np.testing.assert_allclose(other_logp, logp, atol=1e-5, rtol=1e-5)
            self.assertAlmostEqual(other_value, value, places=5)
            for actual, expected in zip(hidden_fast, hidden_eager):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            assert_action_legal(other, legal)

    def test_optimizer_update_refreshes_jit_weights_and_preserves_ppo_audit(self):
        runner = self.runner()
        runner.prepare_inference("jit")
        learner = PPOLearner(runner.network, runner.cfg)
        batch, _ = collect_batch(runner, 1)
        self.assertLess(learner._diagnose(batch, 2)["old_new_logp_max"], 1e-5)
        learner.update(batch)
        runner.prepare_inference("jit")
        next_batch, _ = collect_batch(runner, 1)
        self.assertLess(learner._diagnose(next_batch, 2)["old_new_logp_max"], 1e-5)
        learner.update(next_batch)
        self.assertEqual(learner._total_updates, 2)

    def test_optional_trace_probabilities_match_sampled_action_and_legal_mask(self):
        runner = self.runner()
        runner.capture_action_probabilities = True
        feature = np.random.default_rng(154).normal(size=725).astype(np.float32)
        legal = np.ones(172, dtype=np.float32)
        legal[:12:2] = 0
        legal[12:28:3] = 0
        legal[76:].reshape(12, 8)[:, 0] = 0
        hidden = runner.network.init_hidden(1, "cpu")
        action, logp, _, _ = runner.sample_action(feature, legal, hidden, deterministic=True)
        probabilities = runner.last_head_probabilities
        self.assertEqual([len(head) for head in probabilities], [12, 16, 16, 16, 16, 8])
        for head, values in enumerate(probabilities):
            self.assertAlmostEqual(sum(values), 1.0, places=5)
            self.assertAlmostEqual(values[action[head]] + 1e-5, float(np.exp(logp[head])), places=5)
        self.assertEqual(probabilities[0][0], 0.0)
        self.assertEqual(probabilities[1][0], 0.0)
        self.assertEqual(probabilities[5][0], 0.0)
        assert_action_legal(action, legal)


if __name__ == "__main__":
    unittest.main()
