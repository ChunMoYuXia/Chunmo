"""领域约束不能扩大 SDK 合法集合，且必须能重算真实采样概率。"""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from hok1v1_rl.action_mask import effective_legal_action, assert_action_legal
from hok1v1_rl.behavior_diagnostics import step_evidence
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.matchups import Matchup
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.rollout import EpisodeCollector, join_trajectories
from test_training import FakeGame


class TargetGame(FakeGame):
    def result(self):
        result = super().result()
        for state in result[-1]:
            targets = state["legal_action"][76:].reshape(12, 8)
            targets[3] = 0
            # 交替出现无目标与有小兵目标的情形。
            targets[3, 0] = 1
            if self.index % 2:
                targets[3, 3] = 1
        return result


class EffectivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def state(self, hero=112, target=0):
        legal = np.ones(172, np.float32)
        legal[76:].reshape(12, 8)[3] = 0
        legal[76:].reshape(12, 8)[3, target] = 1
        return {"legal_action": legal, "player_id": 100,
                "req_pb": NS(hero_list=[NS(runtime_id=100, config_id=hero)]),
                "sub_action_mask": {3: [1, 0, 0, 0, 0, 1]}}

    def rule(self):
        cfg = Config().action
        cfg.rule = "targeted_attack_v1"
        return cfg

    def test_null_target_blocks_only_attack_and_never_mutates_sdk(self):
        state = self.state()
        original = state["legal_action"].copy()
        final = effective_legal_action(state, self.rule())
        self.assertEqual(final[3], 0)
        np.testing.assert_array_equal(state["legal_action"], original)
        np.testing.assert_array_equal(np.delete(final, 3), np.delete(original, 3))
        self.assertTrue((final <= original).all())

    def test_hero_soldiers_building_remain_selectable_without_null_or_self(self):
        for target in [1, 3, 4, 5, 6, 7]:
            state = self.state(target=target)
            raw = state["legal_action"][76:].reshape(12, 8)
            raw[3, [0, 2]] = 1
            final = effective_legal_action(state, self.rule())
            row = final[76:].reshape(12, 8)[3]
            self.assertEqual(final[3], 1)
            self.assertEqual(row[target], 1)
            self.assertEqual(row[[0, 2]].sum(), 0)
            np.testing.assert_array_equal(final[76:].reshape(12, 8)[4:], raw[4:])
            self.assertTrue((final <= state["legal_action"]).all())

    def test_special_hero_inactive_target_missing_identity_and_empty_mask_preserved(self):
        for kind in ["special", "inactive", "missing", "only_attack", "empty", "sdk"]:
            state, cfg = self.state(), self.rule()
            if kind == "special":
                state["req_pb"].hero_list[0].config_id = 140
            elif kind == "inactive":
                state["sub_action_mask"][3][5] = 0
            elif kind == "missing":
                state.pop("req_pb")
            elif kind == "only_attack":
                state["legal_action"][:12] = 0
                state["legal_action"][3] = 1
            elif kind == "empty":
                state["legal_action"][:] = 0
            else:
                cfg.rule = "sdk"
            np.testing.assert_array_equal(effective_legal_action(state, cfg), state["legal_action"])

    def test_fragment_eager_and_jit_use_saved_mask_for_ppo_probability_audit(self):
        for mode in ["eager", "jit"]:
            cfg = Config()
            cfg.network.lstm_hidden, cfg.network.lstm_time_steps = 16, 4
            cfg.ppo.minibatch_size = 4
            cfg.action = self.rule()
            network = HoK1v1Network(cfg).eval()
            runner = EnvRunner(TargetGame(end=9), network, cfg, torch.device("cpu"))
            runner.prepare_inference(mode)
            collector = EpisodeCollector(runner, Matchup(112, 132, 1), burn_in=4)
            pieces = [collector.collect(4, version=0) for _ in range(3)]
            batch = join_trajectories(pieces)
            learner = PPOLearner(network, cfg)
            self.assertLess(learner._diagnose(batch, 2)["old_new_logp_max"], 1e-4)
            self.assertLess(learner._diagnose(batch, 2)["old_new_value_max"], 1e-4)
            stats = pieces[-1]["episode_stats"]
            self.assertEqual(stats["state_diagnostics"]["attack_button_blocked_decisions"], 5)
            self.assertEqual(stats["state_diagnostics"]["attack_target_constraint_decisions"], 4)
            self.assertEqual(stats["state_diagnostics"]["attack_without_enemy_target"], 0)
            # 有效掩码保存进 batch；重建 172 维后仍严格属于 SDK 原始合法集合。
            for action, legal in zip(batch["actions"][batch["valid_mask"].bool()],
                                     batch["legal_actions"][batch["valid_mask"].bool()]):
                expanded = np.ones(172, np.float32)
                expanded[:76] = legal[:76].numpy()
                expanded[76:].reshape(12, 8)[int(action[0])] = legal[76:].numpy()
                assert_action_legal(action, expanded)
            learner.update(batch)

    def test_resume_rejects_rule_change_but_new_experiment_records_migration(self):
        cfg = Config()
        cfg.network.lstm_hidden = 16
        source = PPOLearner(HoK1v1Network(cfg), cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            source.save(path)
            changed = copy.deepcopy(cfg)
            changed.action = self.rule()
            target = PPOLearner(HoK1v1Network(changed), changed)
            with self.assertRaisesRegex(ValueError, "action"):
                target.load(path)
            record = target.initialize_from(path)
            self.assertEqual(record["source_action"]["rule"], "sdk")
            self.assertEqual(record["action"]["rule"], "targeted_attack_v1")

    def test_evidence_uses_counters_and_preserves_unknown_without_fabricated_progress(self):
        snap = {"my": {"money": 300, "hurt_to_hero": 100, "hurt_by_hero": 50},
                "en": {"money": 400}, "my_organs": {21: 5000, 24: None},
                "en_organs": {21: 5000, 24: None}}
        after = copy.deepcopy(snap)
        after["my"].update(money=320, hurt_to_hero=170)
        after["en_organs"][21] = 4800
        evidence = step_evidence((3, (0, 0)), (6, (10, 0)), snap, after, 2)
        self.assertEqual(evidence["income_gain"], 20)
        self.assertEqual(evidence["hero_damage_gain"], 70)
        self.assertEqual(evidence["enemy_building_damage"], 200)
        self.assertEqual(evidence["move_observed_displacements"], 1)
        missing = step_evidence((None, None), (None, None), None, None, 2)
        self.assertEqual(missing["snapshot_step_count"], 0)
        self.assertEqual(missing["move_position_known_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
