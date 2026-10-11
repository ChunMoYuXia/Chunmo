"""覆盖真实历史失败模式，全部使用 CPU，无需 gamecore。"""
import copy
import importlib.util
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hok1v1_rl.action_mask import compress_legal_action, masked_softmax
from hok1v1_rl.config import Config, HERO_IDS
from hok1v1_rl.env_runner import EnvRunner, state_game_ids
from hok1v1_rl.evaluation import summarize_games, evaluate, wilson_interval
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.matchups import Matchup, TrainingSchedule, evaluation_schedule
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.policy_diagnostics import PolicyDiagnostics, merge_policy_diagnostics
from hok1v1_rl.reward import compute_value_targets, determine_outcome, extract_hero_snapshot, organ_damage
from hok1v1_rl.reward import state_reward_terms
from hok1v1_rl.rollout import collect_batch, near_fountain


class FakeGame:
    """提供与 SDK 相同的 reset/step 返回结构；状态随动作推进，末步摧毁水晶。"""
    addrs = []
    runtime_id = "test"
    def __init__(self, end=5, unknown=False):
        self.end, self.unknown = end, unknown
        self.player_list = [100, 200]
        self.game_launcher = NS(stop_game=lambda _: None, check_exists_game=lambda _: False)
        self.resets = []
        self.native_eval_flags = []

    def reset(self, camp, use_common_ai, eval):
        self.resets.append(camp)
        self.native_eval_flags.append(eval)
        self.side = use_common_ai.index(False)
        self.player_masks = use_common_ai.copy()
        self.heroes = [camp["heroes"][i][0]["hero_id"] for i in (0, 1)]
        self.index = 0
        return self.result()

    def step(self, actions):
        self.index += 1
        return self.result()

    def result(self):
        done = self.index >= self.end
        heroes = [NS(camp=i+1, config_id=self.heroes[i], runtime_id=100+i*100,
                     hp=4000, max_hp=5000, money=300-self.index,
                     moneyCnt=300+self.index*20, exp=self.index*10, ep=100,
                     killCnt=0, deadCnt=0, totalHurtToHero=self.index*5,
                     totalBeHurtByHero=self.index*2, camp_visible=[True, True]) for i in (0, 1)]
        organs = [NS(camp=i+1, type=kind,
                     hp=0 if done and not self.unknown and i != self.side and kind == 24 else 5000)
                  for i in (0, 1) for kind in (21, 24)]
        req = NS(hero_list=heroes, organ_list=organs, frame_no=self.index*3)
        states = [{"observation": np.full(705, self.index / 100, np.float32),
                   "game_id": f"fake-game-{len(self.resets)}", "sgame_id": len(self.resets),
                   "player_id": 100+i*100, "req_pb": req, "legal_action": np.ones(172, np.float32),
                   "sub_action_mask": {button: [1,1,1,0,0,1] for button in range(12)}} for i in (0, 1)]
        return None, None, [done, done], states


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        cls.cfg = Config()
        cls.cfg.network.lstm_hidden = 64
        cls.cfg.network.lstm_time_steps = 4
        cls.cfg.env.hero_ids = [112, 169]
        cls.cfg.env.opponent_hero_ids = [132, 133]
        cls.cfg.ppo.minibatch_size = 4
        cls.net = HoK1v1Network(cls.cfg)

    def runner(self, end=5, unknown=False):
        return EnvRunner(FakeGame(end, unknown), copy.deepcopy(self.net), copy.deepcopy(self.cfg), torch.device("cpu"))

    def test_mask_and_conditional_target(self):
        probs = masked_softmax(torch.randn(12), torch.tensor([1,1]+[0]*10))
        self.assertEqual(float(probs[2:].sum()), 0)
        legal = torch.arange(172).float()
        compressed = compress_legal_action(legal, torch.tensor(3))
        torch.testing.assert_close(compressed[-8:], legal[76:].reshape(12,8)[3])

    def test_income_uses_cumulative_counter(self):
        req = NS(hero_list=[NS(camp=1, money=20, moneyCnt=700, camp_visible=[1,1])])
        self.assertEqual(extract_hero_snapshot(req,1)["money"],700)

    def test_sdk_ids_are_decoded_without_claiming_abs_filename_id(self):
        ids = state_game_ids({"game_id": b"client-148", "sgame_id": "client-148"})
        self.assertEqual(ids, {"sdk_game_id": "client-148", "sdk_client_id": "client-148"})

    def test_skill_resource_consumption_gives_no_reward(self):
        runner = self.runner()
        runner.reset(matchup=Matchup(112, 133, 0))
        previous = copy.deepcopy(runner._last_snapshot)
        current = copy.deepcopy(previous)
        current["my"]["ep"] = 0
        self.assertEqual(sum(state_reward_terms(previous, current).values()), 0)
        current["my"]["hurt_to_hero"] += 500
        self.assertGreater(state_reward_terms(previous, current)["hp"], 0)

    def test_fountain_diagnostic_uses_position_and_excludes_dead_or_missing(self):
        hero = NS(camp=2, hp=100, location=NS(x=37000, z=38000))
        fountain = NS(camp=2, type=23, location=NS(x=38000, z=38000))
        state = {"req_pb": NS(hero_list=[hero], organ_list=[fountain])}
        self.assertTrue(near_fountain(state, 2))
        hero.location.x = 0
        self.assertFalse(near_fountain(state, 2))
        hero.hp = 0
        self.assertIsNone(near_fountain(state, 2))
        hero.hp, hero.location = 100, None
        self.assertIsNone(near_fountain(state, 2))

    def test_control_identity_and_timestamps_on_both_sides(self):
        from datetime import datetime
        for side in (0, 1):
            runner = self.runner(end=1)
            # 双方相同英雄也必须能区分实际受控的 runtime_id 和 camp。
            game = runner.collect_trajectory(matchup=Matchup(112, 112, side))["episode_stats"]
            self.assertEqual(game["control"]["runtime_player_id"], 100+side*100)
            self.assertEqual(game["control"]["camp"], side+1)
            self.assertEqual(game["control"]["common_ai"], [i != side for i in range(2)])
            self.assertTrue(game["control"]["verified"])
            self.assertLessEqual(datetime.fromisoformat(game["started_at"]), datetime.fromisoformat(game["ended_at"]))
            self.assertEqual(game["sdk_game_id"], "fake-game-1")
            self.assertEqual(game["outcome_evidence"]["control"], game["control"])

    def test_control_mismatch_rejects_wrong_player_hero_or_ai_mask(self):
        for error in ("player", "hero", "mask"):
            runner = self.runner()
            original = runner.env.reset
            def bad_reset(*args, **kwargs):
                result = original(*args, **kwargs)
                state = result[3][runner.player_id]
                if error == "player":
                    state["player_id"] = 200 if runner.player_id == 0 else 100
                elif error == "hero":
                    state["req_pb"].hero_list[runner.player_id].config_id = 169
                else:
                    runner.env.need_predict = lambda i: i != runner.player_id
                return result
            runner.env.reset = bad_reset
            with self.assertRaisesRegex(RuntimeError, "控制方校验失败"):
                runner.reset(matchup=Matchup(112, 133, 0))

    def test_control_change_midgame_rejects_before_reward(self):
        runner = self.runner()
        runner.reset(matchup=Matchup(112, 133, 0))
        original = runner.env.step
        def changed_game(actions):
            result = original(actions)
            result[3][0]["game_id"] = "another-game"
            return result
        runner.env.step = changed_game
        with self.assertRaisesRegex(RuntimeError, "game_id"):
            runner.step([1,0,0,0,0,0])

    def test_natural_game_end_waits_for_replay_flush_without_force_stop(self):
        runner = self.runner(end=1)
        runner.collect_trajectory(matchup=Matchup(112, 133, 0))
        checks = iter([True, False])
        launcher = runner.env.game_launcher
        launcher.check_exists_game = lambda _: next(checks)
        with patch.object(launcher, "stop_game") as stop, patch("hok1v1_rl.env_runner.time.sleep") as sleep:
            runner.close()
        stop.assert_not_called()
        sleep.assert_called_once_with(.1)
        self.assertFalse(runner._game_done)

    def test_slow_natural_exit_does_not_abort_after_ten_seconds(self):
        runner = self.runner(end=1)
        runner.collect_trajectory(matchup=Matchup(112, 133, 0))
        clock = [0.0]
        launcher = runner.env.game_launcher
        launcher.check_exists_game = lambda _: clock[0] < 12.0
        with patch.object(launcher, "stop_game") as stop, \
                patch("hok1v1_rl.env_runner.time.monotonic", side_effect=lambda: clock[0]), \
                patch("hok1v1_rl.env_runner.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0]+seconds)):
            runner.close()
        stop.assert_not_called()
        self.assertGreaterEqual(clock[0], 12.0)
        self.assertFalse(runner._game_done or runner._game_running)

    def test_natural_exit_timeout_stops_old_game_and_allows_next_game(self):
        runner = self.runner(end=1)
        runner.collect_trajectory(matchup=Matchup(112, 133, 0))
        clock, stopped_at = [0.0], [None]
        launcher = runner.env.game_launcher
        launcher.check_exists_game = lambda _: stopped_at[0] is None or clock[0] < stopped_at[0]+.2
        with patch.object(launcher, "stop_game", side_effect=lambda _: stopped_at.__setitem__(0, clock[0])) as stop, \
                patch("hok1v1_rl.env_runner.time.monotonic", side_effect=lambda: clock[0]), \
                patch("hok1v1_rl.env_runner.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0]+seconds)), \
                self.assertLogs("hok1v1_rl.env_runner", level="WARNING") as logs:
            runner.reset(matchup=Matchup(112, 133, 1))
        stop.assert_called_once_with("test")
        self.assertGreaterEqual(stopped_at[0], 30.0)
        self.assertEqual(len(runner.env.resets), 2)
        self.assertIn("回放可能不完整", logs.output[0])
        self.assertTrue(runner._game_running)

    def test_stop_timeout_prevents_starting_a_new_game(self):
        runner = self.runner(end=1)
        runner.collect_trajectory(matchup=Matchup(112, 133, 0))
        clock = [0.0]
        launcher = runner.env.game_launcher
        launcher.check_exists_game = lambda _: True
        with patch.object(launcher, "stop_game") as stop, \
                patch("hok1v1_rl.env_runner.time.monotonic", side_effect=lambda: clock[0]), \
                patch("hok1v1_rl.env_runner.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0]+seconds)), \
                self.assertLogs("hok1v1_rl.env_runner", level="WARNING"), \
                self.assertRaisesRegex(RuntimeError, "stop_game 后 gamecore 仍未退出"):
            runner.reset(matchup=Matchup(112, 133, 1))
        stop.assert_called_once_with("test")
        self.assertEqual(len(runner.env.resets), 1)
        self.assertTrue(runner._game_running)
        self.assertFalse(runner._game_done)

    def test_selection_frequency_is_not_probability(self):
        diagnostic = PolicyDiagnostics()
        legal = np.zeros(172)
        legal[1:3] = 1
        probability = np.zeros(12)
        probability[1:3] = [.501, .499]
        for _ in range(10):
            diagnostic.add(legal, probability, 1)
        result = diagnostic.to_dict()
        self.assertEqual(result["selection_rate"][1], 1.0)
        self.assertAlmostEqual(result["probability_mean"][1], .501)
        self.assertEqual(result["near_tie_rate"], 1.0)
        self.assertIsNone(result["probability_when_legal"][3])
        other = PolicyDiagnostics()
        other.add(legal, probability, 2)
        merged = merge_policy_diagnostics([{"policy_diagnostics": result},
                                           {"policy_diagnostics": other.to_dict()}])
        self.assertAlmostEqual(merged["selection_rate"][1], 10 / 11)

    def test_gradient_diagnosis_preserves_model_optimizer_and_mode(self):
        runner = self.runner()
        batch, _ = collect_batch(runner, 5)
        batch["returns"] = batch["returns"] + 100
        learner = PPOLearner(runner.network, runner.cfg)
        network = runner.network.eval()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        result = learner.diagnose_gradients(batch, torch.arange(len(batch["features"])))
        self.assertFalse(network.training)
        self.assertFalse(learner.optimizer.state)
        self.assertTrue(all(p.grad is None for p in network.parameters()))
        self.assertGreater(result["value_grad_norm"], result["policy_grad_norm"])
        self.assertLess(result["estimated_global_clip_scale"], .1)
        for key, value in before.items():
            torch.testing.assert_close(value, network.state_dict()[key])

    def test_weights_only_experiment_allows_ppo_changes(self):
        runner = self.runner()
        learner = PPOLearner(runner.network, runner.cfg)
        batch, _ = collect_batch(runner, 5)
        learner.update(batch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.pt"
            learner.save(path, {"env_steps": 100})
            changed = copy.deepcopy(runner.cfg)
            changed.ppo.ppo_epochs = 4
            experiment = PPOLearner(copy.deepcopy(self.net), changed)
            rng = torch.get_rng_state().clone()
            metadata = experiment.initialize_from(path)
            self.assertEqual(metadata["source_updates"], 1)
            self.assertEqual(experiment._total_updates, 0)
            self.assertFalse(experiment.optimizer.state)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            for key, value in runner.network.state_dict().items():
                torch.testing.assert_close(experiment.network.state_dict()[key], value)
            with self.assertRaisesRegex(ValueError, "ppo"):
                experiment.load(path)

    def test_separate_clipping_protects_policy_head_from_large_value_gradient(self):
        learner = PPOLearner(copy.deepcopy(self.net), copy.deepcopy(self.cfg))
        groups = learner.gradient_groups
        grouped = [id(p) for group in groups for p in group]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in learner.network.parameters()})
        def assign_gradients():
            for p in learner.network.parameters():
                p.grad = torch.zeros_like(p)
            groups[0][0].grad.flatten()[0] = .1
            groups[1][0].grad.flatten()[0] = 100
        assign_gradients()
        _, global_scale = learner._clip_gradients()
        self.assertLess(global_scale, .01)
        self.assertLess(float(groups[0][0].grad.norm()), .001)
        assign_gradients()
        learner.cfg.ppo.grad_clip_mode = "separate"
        _, separate_scale = learner._clip_gradients()
        self.assertEqual(separate_scale, 1)
        self.assertAlmostEqual(float(groups[0][0].grad.norm()), .1, places=6)
        self.assertLessEqual(float(groups[1][0].grad.norm()), learner.cfg.ppo.max_grad_norm)

    def test_resume_accepts_checkpoint_before_gradient_mode_field(self):
        learner = PPOLearner(copy.deepcopy(self.net), copy.deepcopy(self.cfg))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-v2.pt"
            learner.save(path)
            state = torch.load(path, weights_only=False)
            del state["config"]["ppo"]["grad_clip_mode"]
            torch.save(state, path)
            learner.load(path)
            self.assertEqual(learner.cfg.ppo.grad_clip_mode, "global")

    def test_old_repeat_checkpoint_requires_explicit_new_experiment(self):
        learner = PPOLearner(copy.deepcopy(self.net), copy.deepcopy(self.cfg))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-repeat.pt"
            learner.save(path)
            state = torch.load(path, weights_only=False)
            for contract in (None, "fresh-remote-control-v1"):
                if contract is None:
                    state.pop("environment_contract", None)
                else:
                    state["environment_contract"] = contract
                torch.save(state, path)
                with self.assertRaisesRegex(ValueError, "--init-from"):
                    learner.load(path)
            self.assertEqual(learner.initialize_from(path)["mode"], "weights_only")

    def test_live_progress_does_not_change_trajectory(self):
        runner = self.runner()
        updates = []
        runner.progress_callback = lambda **data: updates.append(data)
        trajectory = runner.collect_trajectory(deterministic=True)
        self.assertEqual(updates[0]["episode_decisions"], 0)
        self.assertIn("hero_id", updates[0])
        self.assertEqual(updates[-1]["episode_decisions"], trajectory["episode_len"])
        self.assertEqual(len(updates), trajectory["episode_len"] + 1)
        runner.close()

    def test_training_entry_writes_live_status_and_evaluation(self):
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("train_entry", root / "scripts/train_ppo.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            source = Path(tmp) / "source.pt"
            parent = PPOLearner(copy.deepcopy(self.net), copy.deepcopy(self.cfg))
            parent._total_updates = 66
            parent.save(source)
            changed = copy.deepcopy(self.cfg)
            changed.ppo.grad_clip_mode = "separate"
            args = NS(config=None, resume=None, init_from=str(source), total_steps=1, device="cpu", runtime_id=None,
                      zmq_port=None, gamecore_addr=None, log_dir=str(directory), max_decisions=None,
                      batch_size=8, minibatch_size=4, eval_interval=1, eval_episodes=2)
            tb = NS(log_metrics=lambda *args: None, close=lambda: None)
            with patch.object(entry, "parse_args", return_value=args), \
                 patch.object(entry, "load_config", return_value=changed), \
                 patch.object(entry, "build_hok_env", return_value=FakeGame()), \
                 patch.object(entry, "UnifiedLogger", return_value=tb):
                entry.main()
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["phase"], "finished")
            self.assertEqual(status["completed_updates"], 1)
            self.assertEqual(status["eval_completed"], 2)
            self.assertEqual(status["episode_decisions"], 5)
            self.assertEqual(status["env_steps"], 10)
            rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual([r["mode"] for r in rows], ["train", "train", "eval", "eval"])
            self.assertTrue(all("timestamp" in r for r in rows))
            self.assertTrue((directory / "final_ckpt.pt").exists())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["initialization"]["source_updates"], 66)
            self.assertEqual(manifest["initialization"]["mode"], "weights_only")
            self.assertEqual(manifest["config"]["ppo"]["grad_clip_mode"], "separate")
            metrics = json.loads((directory / "metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("policy/button_legal_rate/3", metrics)
            self.assertIn("train/policy_head_clip_scale", metrics)
            self.assertGreater(metrics["res/update_seconds"], 0)

    def test_missing_building_has_no_reward(self):
        self.assertEqual(organ_damage({21:5000,24:7000}, {21:None,24:None}),0)
        self.assertEqual(organ_damage({21:None,24:None}, {21:5000,24:7000}),0)
        self.assertEqual(organ_damage({21:5000,24:7000}, {21:0,24:7000}),5000)

    def test_unknown_time_limit_is_not_invented_win(self):
        req = NS(organ_list=[NS(camp=1,type=24,hp=6000),NS(camp=2,type=24,hp=1000)])
        self.assertIsNone(determine_outcome(req,1,2))
        req.organ_list[1].hp=0
        self.assertEqual(determine_outcome(req,1,2),"win")

    def test_terminal_gae_and_external_truncation(self):
        _, terminal = compute_value_targets([1],.99,[5],100,.95,[1])
        _, truncated = compute_value_targets([1],.99,[5],100,.95,[0])
        np.testing.assert_allclose(terminal,[1])
        np.testing.assert_allclose(truncated,[100])
        runner=self.runner()
        trajectory=runner.collect_trajectory(max_decisions=2)
        self.assertEqual(float(trajectory["dones"].sum()),0)
        stats=trajectory["episode_stats"]
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["reward_components"]["terminal"],0)
        self.assertAlmostEqual(float(trajectory["returns"][0,1]),
                               float(trajectory["rewards"][0,1])+runner.cfg.ppo.gamma*stats["bootstrap_value"],places=5)
        runner._game_running=False

    def test_terminal_sample_and_component_sum_are_preserved(self):
        runner=self.runner()
        trajectory=runner.collect_trajectory()
        self.assertEqual(trajectory["features"].shape[:2],(2,4))
        self.assertEqual(float(trajectory["valid_mask"].sum()),5)
        self.assertEqual(float(trajectory["dones"][1,0]),1)
        stats=trajectory["episode_stats"]
        self.assertEqual(stats["reward_components"]["terminal"],50)
        self.assertAlmostEqual(stats["reward"],sum(stats["reward_components"].values()))
        self.assertGreater(stats["reward_components"]["money"],0)
        self.assertAlmostEqual(stats["reward_components"]["money"],0.05)
        self.assertEqual(stats["state_diagnostics"]["wallet_decreases"],5)
        self.assertEqual(stats["state_diagnostics"]["income_counter_decreases"],0)

    def test_sequential_lstm_equals_sequence_replay(self):
        net=self.net.eval()
        features=torch.randn(2,7,725)
        initial=net.init_hidden(2,"cpu")
        whole=net(features,initial)
        hidden=initial
        parts=[]
        for index in range(7):
            out=net(features[:,index:index+1],hidden)
            hidden=out["lstm_state_new"]
            parts.append(out["logits_list"])
        for head in range(6):
            torch.testing.assert_close(whole["logits_list"][head],torch.cat([p[head] for p in parts],1),atol=2e-5,rtol=2e-5)

    def loss_args(self, batch):
        return dict(features=batch["features"], legal_actions=batch["legal_actions"], actions=batch["actions"],
                    old_log_probs=batch["log_probs"], advantages=batch["advantages"], returns=batch["returns"],
                    old_values=batch["values"], head_weights=batch["head_weights"], valid_mask=batch["valid_mask"],
                    lstm_state_init=batch["lstm_state_init"])

    def test_inactive_heads_and_padding_do_not_change_kl(self):
        runner=self.runner()
        batch,_=collect_batch(runner,5)
        learner=PPOLearner(runner.network,runner.cfg)
        args=self.loss_args(batch)
        base=learner.compute_loss(**args)
        changed=batch["log_probs"].clone()
        changed[...,3:5]+=1000
        changed[~batch["valid_mask"].bool()]+=1000
        args["old_log_probs"]=changed
        result=learner.compute_loss(**args)
        for key in ("loss","kl","entropy","clip_fraction","old_new_logp_gap"):
            torch.testing.assert_close(base[key],result[key])
        args["old_log_probs"]=changed.clone()
        args["old_log_probs"][...,0]+=0.2*batch["valid_mask"]
        self.assertGreater(learner.compute_loss(**args)["kl"].item(),0.001)

    def test_update_changes_weights_and_resume_restores_state(self):
        runner=self.runner()
        batch,_=collect_batch(runner,5)
        learner=PPOLearner(runner.network,runner.cfg)
        before={k:v.clone() for k,v in runner.network.state_dict().items()}
        result=learner.update(batch)
        self.assertGreater(result.num_minibatches,0)
        self.assertLess(result.old_new_logp_gap,1e-3)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in runner.network.state_dict().items()))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"model.pt"
            learner.save(path,{"env_steps":5,"schedule":runner.schedule.state_dict()})
            expected=(random.random(),np.random.rand(),torch.rand(2))
            restored=PPOLearner(copy.deepcopy(self.net),copy.deepcopy(self.cfg))
            state=restored.load(path)
            self.assertEqual(state["env_steps"],5)
            self.assertEqual(restored._total_updates,1)
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(2),expected[2])
            for key,value in runner.network.state_dict().items():
                torch.testing.assert_close(restored.network.state_dict()[key],value)
            self.assertTrue(restored.optimizer.state)

    def test_invalid_old_prob_rejects_before_parameter_update(self):
        runner=self.runner()
        batch,_=collect_batch(runner,5)
        batch["log_probs"][...,0]+=1
        learner=PPOLearner(runner.network,runner.cfg)
        before=[p.clone() for p in runner.network.parameters()]
        with self.assertRaises(RuntimeError): learner.update(batch)
        for old,new in zip(before,runner.network.parameters()): torch.testing.assert_close(old,new)

    def test_padding_is_excluded_from_value_metrics(self):
        runner=self.runner()
        batch,_=collect_batch(runner,5)
        learner=PPOLearner(runner.network,runner.cfg)
        baseline=learner._diagnose(batch,1)
        changed=copy.deepcopy(batch)
        padded=~changed["valid_mask"].bool()
        changed["returns"][padded]=1e6
        changed["advantages"][padded]=1e6
        result=learner._diagnose(changed,1)
        for key in ("value_loss","explained_variance","policy_loss","kl"):
            self.assertAlmostEqual(baseline[key],result[key],places=6)

    def test_resume_rejects_changed_task_and_legacy_model(self):
        runner=self.runner()
        learner=PPOLearner(runner.network,runner.cfg)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"model.pt"
            learner.save(path)
            changed=copy.deepcopy(runner.cfg)
            changed.env.hero_ids=[112]
            restored=PPOLearner(copy.deepcopy(self.net),changed)
            with self.assertRaises(ValueError): restored.load(path)
            torch.save({"network":self.net.state_dict()},path)
            with self.assertRaisesRegex(ValueError,"旧 checkpoint"): learner.load(path)

    def test_kl_stops_entire_update_before_step(self):
        runner=self.runner()
        batch,_=collect_batch(runner,5)
        learner=PPOLearner(runner.network,runner.cfg)
        original=learner._loss
        calls=0
        def excessive(batch,indices):
            nonlocal calls
            losses=original(batch,indices)
            calls+=1
            if calls>2: losses["kl"]=torch.tensor(1.0)
            return losses
        learner._loss=excessive  # 前两次是全 batch 前置审计，随后模拟 KL 超限
        before=[p.clone() for p in runner.network.parameters()]
        result=learner.update(batch)
        self.assertTrue(result.early_stopped)
        self.assertEqual(result.num_minibatches,0)
        for old,new in zip(before,runner.network.parameters()): torch.testing.assert_close(old,new)

    def test_balanced_random_heroes_and_checkpoint_schedule(self):
        schedule=TrainingSchedule(HERO_IDS,HERO_IDS,42)
        first=[schedule.next() for _ in range(40)]
        self.assertEqual({(m.hero_id,m.player_id) for m in first},{(h,s) for h in HERO_IDS for s in (0,1)})
        saved=schedule.state_dict()
        expected=schedule.next()
        schedule.load_state_dict(saved)
        self.assertEqual(expected,schedule.next())
        full=evaluation_schedule(HERO_IDS,HERO_IDS,1,800)
        self.assertEqual(len(set(full)),800)
        self.assertEqual(full,evaluation_schedule(HERO_IDS,HERO_IDS,1,800))

    def test_true_eval_mode_preserves_training_rng_and_schedule(self):
        runner=self.runner(end=1)
        schedule=runner.schedule.state_dict()
        rng=torch.get_rng_state().clone()
        summary=evaluate(runner,4)
        self.assertEqual(runner.env.native_eval_flags,[True]*4)
        self.assertEqual(runner.schedule.state_dict(),schedule)
        torch.testing.assert_close(torch.get_rng_state(),rng)
        self.assertEqual(summary["wins"],4)
        self.assertFalse(summary["goal_verified"])

    def test_conservative_win_denominator(self):
        games=[dict(hero_id=112,player_id=0,win=i<9,outcome="win" if i<9 else "unknown",truncated=i==9) for i in range(10)]
        result=summarize_games(games)
        self.assertEqual(result["win_rate"],.9)
        self.assertFalse(result["goal_verified"])
        self.assertLess(wilson_interval(9,10)[0],.9)

    def test_bad_config_fails_fast(self):
        with self.assertRaises(ValueError): Config.from_dict({"ppo":{"batch_szie":16}})
        with self.assertRaises(ValueError): Config.from_dict({"env":{"num_envs":2}})
        with self.assertRaises(FileNotFoundError): Config.from_yaml("missing-config.yaml")

if __name__ == "__main__":
    unittest.main()
