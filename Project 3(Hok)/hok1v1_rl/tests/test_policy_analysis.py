"""动作信用和探索诊断必须排除 padding、不生效头，并保持 learner 状态。"""
import copy
import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.learner import PPOLearner, normalize_advantages
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.policy_analysis import action_advantages, head_exploration
from hok1v1_rl.rollout import collect_batch
from hok1v1_rl.reward import compute_value_targets
from test_training import FakeGame


class AlternatingGame(FakeGame):
    def result(self):
        result = super().result()
        for state in result[-1]:
            state["legal_action"][:12] = 0
            state["legal_action"][1 if self.index % 2 == 0 else 2] = 1
            for hero in state["req_pb"].hero_list:
                hero.location = NS(x=self.index * 10, z=0)
        return result


class PolicyAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self):
        cfg = Config()
        cfg.network.lstm_hidden, cfg.network.lstm_time_steps = 16, 4
        cfg.ppo.minibatch_size = 8
        network = HoK1v1Network(cfg)
        runner = EnvRunner(FakeGame(end=5), network, cfg, torch.device("cpu"))
        batch, _ = collect_batch(runner, 5)
        return PPOLearner(network, cfg), batch

    def test_advantage_diagnostic_is_normalization_invariant_and_padding_independent(self):
        _, batch = self.batch()
        expected = action_advantages(batch)
        self.assertEqual(expected["valid_decisions"], 5)
        self.assertEqual(sum(v["decisions"] for v in expected["by_button"].values()), 5)
        self.assertEqual(sum(sum(v["decisions"] for v in h.values()) for h in expected["by_hero"].values()), 5)
        self.assertEqual(expected, action_advantages(normalize_advantages(batch)))
        changed = copy.deepcopy(batch)
        padding = ~batch["valid_mask"].bool()
        for key in ["advantages", "returns", "values", "features", "head_weights", "legal_actions"]:
            changed[key][padding] = 1e6
        self.assertEqual(expected, action_advantages(changed))

    def test_entropy_excludes_inactive_heads_and_uses_full_batch(self):
        learner, batch = self.batch()
        batch["head_weights"].zero_()
        batch["head_weights"][..., 0] = 1
        batch["legal_actions"][..., :12] = 0
        batch["legal_actions"][..., 1] = 1
        result = head_exploration(learner, batch)
        self.assertEqual(result["0"]["active_decisions"], 5)
        self.assertAlmostEqual(result["0"]["top1_probability_mean"], 1)
        self.assertLess(abs(result["0"]["entropy_mean"]), 2e-5)
        self.assertIsNone(result["0"]["normalized_entropy_mean"])
        self.assertTrue(all(result[str(i)]["active_decisions"] == 0 for i in range(1, 6)))

    def test_split_gradients_preserve_model_rng_mode_and_optimizer(self):
        learner, batch = self.batch()
        learner.network.eval()
        before = {k: v.clone() for k, v in learner.network.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = learner.diagnose_gradients(batch, torch.arange(len(batch["features"])), separate_entropy=True)
        self.assertGreater(result["surrogate_grad_norm"], 0)
        self.assertGreater(result["entropy_grad_norm"], 0)
        self.assertFalse(learner.network.training)
        self.assertFalse(learner.optimizer.state)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in learner.network.parameters()))
        for k, v in before.items():
            torch.testing.assert_close(v, learner.network.state_dict()[k])
        learner.cfg.ppo.entropy_beta_start = learner.cfg.ppo.entropy_beta_end = 0
        result = learner.diagnose_gradients(batch, torch.arange(len(batch["features"])), separate_entropy=True)
        self.assertLess(result["entropy_grad_norm"], 1e-6)

    def test_behavior_counters_cannot_replace_value_predictions_or_gae(self):
        for mode in ("eager", "jit"):
            cfg = Config()
            cfg.network.lstm_hidden, cfg.network.lstm_time_steps = 16, 4
            cfg.ppo.minibatch_size = 8
            network = HoK1v1Network(cfg).eval()
            with torch.no_grad():
                network.value_head[-1].weight.zero_()
                network.value_head[-1].bias.fill_(7.25)
            runner = EnvRunner(AlternatingGame(end=5), network, cfg, torch.device("cpu"))
            runner.prepare_inference(mode)
            batch, _ = collect_batch(runner, 5)
            valid = batch["valid_mask"].bool()
            self.assertEqual(set(batch["actions"][valid][:, 0].tolist()), {1, 2})
            torch.testing.assert_close(batch["values"][valid], torch.full((5,), 7.25))
            advantages, returns = compute_value_targets(
                batch["rewards"][valid].numpy(), cfg.ppo.gamma, np.full(5, 7.25),
                0, cfg.ppo.lamda, batch["dones"][valid].numpy())
            np.testing.assert_allclose(batch["advantages"][valid], advantages, atol=1e-5)
            np.testing.assert_allclose(batch["returns"][valid], returns, atol=1e-5)
            self.assertLess(PPOLearner(network, cfg)._diagnose(batch, 2)["old_new_value_max"], 1e-5)

    def test_value_corruption_is_rejected_before_optimizer_update(self):
        learner, batch = self.batch()
        before = {k: v.clone() for k, v in learner.network.state_dict().items()}
        batch["values"][batch["valid_mask"].bool()] += 1
        with self.assertRaisesRegex(RuntimeError, "value"):
            learner.update(batch)
        self.assertFalse(learner.optimizer.state)
        for key, value in before.items():
            torch.testing.assert_close(value, learner.network.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
