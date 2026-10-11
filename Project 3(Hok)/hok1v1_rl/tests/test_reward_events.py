"""过程奖励的真实事件、未知证据、片段守恒及奖励迁移。"""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from hok1v1_rl.config import Config
from hok1v1_rl.env_runner import EnvRunner
from hok1v1_rl.learner import PPOLearner
from hok1v1_rl.matchups import Matchup
from hok1v1_rl.network import HoK1v1Network
from hok1v1_rl.reward import extract_hero_snapshot, sdk_last_hit, state_reward_terms, terminal_reward_terms
from hok1v1_rl.reward_diagnostics import step_reward_detection, summarize_reward_detection
from hok1v1_rl.rollout import EpisodeCollector, join_trajectories
from test_training import FakeGame


class EventGame(FakeGame):
    def result(self):
        result = super().result()
        for state in result[-1]:
            state["reward"] = [0.] * 10
            state["reward"][5] = (1 if self.index % 2 else -1) if self.index else 0
            state["req_pb"].hero_list[1].moneyCnt = 300 + self.index * 12
        return result


class RewardEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cfg(self):
        cfg = Config.from_yaml(Path(__file__).resolve().parents[1] / 'configs/reward_events.yaml')
        cfg.network.lstm_hidden, cfg.network.lstm_time_steps = 16, 4
        return cfg

    def snapshots(self):
        mine = dict(money=1000., wallet=1000., hp=5000., max_hp=5000., hurt_to_hero=100.,
                    hurt_by_hero=100., kills=0., deaths=0., exp=0.)
        enemy = dict(money=2000., hp=5000., max_hp=5000., visible=False)
        previous = dict(my=mine, en=enemy, my_organs={21:5000}, en_organs={21:5000},
                        frame_no=10, sdk_last_hit=0., sdk_last_hit_status='available')
        current = copy.deepcopy(previous)
        current['frame_no'] = 13
        return previous, current

    def test_comeback_gets_credit_wallet_spending_and_static_lead_do_not(self):
        a, b = self.snapshots()
        b['my'].update(money=1500., wallet=0.)
        b['en']['money'] = 2100.
        self.assertAlmostEqual(state_reward_terms(a,b,self.cfg().reward)['economy_diff'], .2)
        a['my']['money'] = b['my']['money'] = 3000.
        a['en']['money'] = b['en']['money'] = 2000.
        self.assertEqual(state_reward_terms(a,b,self.cfg().reward)['economy_diff'], 0.)
        b['my']['money'] = 20.  # 回绕不应伪造一次巨额负奖励。
        self.assertEqual(state_reward_terms(a,b,self.cfg().reward)['economy_diff'], 0.)
        self.assertEqual(step_reward_detection(a,b,1)['economy_available_steps'], 0)

    def test_damage_uses_counters_not_fog_hp_and_resource_use_is_not_rewarded(self):
        a,b=self.snapshots()
        b['en']['hp']=1.
        self.assertEqual(sum(state_reward_terms(a,b,self.cfg().reward).values()), 0.)
        b['my'].update(hurt_to_hero=1100.,hurt_by_hero=600.,ep=0.)
        terms=state_reward_terms(a,b,self.cfg().reward)
        self.assertAlmostEqual(terms['damage_dealt'],.2)
        self.assertAlmostEqual(terms['damage_taken'],-.075)
        self.assertEqual(terms['hp'],0.)
        b['my']['hurt_to_hero']=None
        self.assertEqual(state_reward_terms(a,b,self.cfg().reward)['damage_dealt'],0.)

    def test_sdk_net_last_hit_is_raw_signed_and_never_counts_missing_as_zero_evidence(self):
        raw=[0.] * 10; raw[5]=2.
        self.assertEqual(sdk_last_hit({'reward':raw}),(2.,'available'))
        a,b=self.snapshots(); b['sdk_last_hit']=2.
        self.assertAlmostEqual(state_reward_terms(a,b,self.cfg().reward)['last_hit'],.04)
        b['sdk_last_hit']=-1.
        self.assertAlmostEqual(state_reward_terms(a,b,self.cfg().reward)['last_hit'],-.02)
        self.assertEqual(step_reward_detection(a,b,3)['last_hit_negative_credits'],1)
        for raw in ([0.] * 9,[0.] * 5+[.5]+[0.] * 4,[0.] * 5+[float('nan')]+[0.] * 4):
            self.assertEqual(sdk_last_hit({'reward':raw}),(None,'invalid'))
        self.assertEqual(sdk_last_hit({}),(None,'missing'))
        b['frame_no']=a['frame_no']
        self.assertEqual(state_reward_terms(a,b,self.cfg().reward)['last_hit'],0.)
        b['sdk_last_hit']=None; b['sdk_last_hit_status']='missing'
        evidence=step_reward_detection(a,b,3)
        self.assertEqual(evidence['last_hit_available_steps'],0)
        self.assertEqual(evidence['last_hit_missing_steps'],1)

    def test_hit_records_are_diagnostics_and_not_extra_damage_or_dodge_rewards(self):
        enemy=NS(camp=2,runtime_id=20)
        mine=NS(camp=1,runtime_id=10,hit_target_info=[NS(hit_target=[20],slot_type=0),
                    NS(hit_target=[20],slot_type=2)])
        req=NS(hero_list=[mine,enemy])
        snap=extract_hero_snapshot(req,1)
        self.assertEqual(snap['hero_hit_records'],2)
        self.assertEqual(snap['hero_skill_hit_records'],1)
        a,b=self.snapshots(); b['my'].update(snap)
        self.assertEqual(state_reward_terms(a,b,self.cfg().reward)['damage_dealt'],0.)

    def test_speed_bonus_only_on_known_true_win(self):
        weights=self.cfg().reward
        self.assertEqual(terminal_reward_terms('win',5100,100,20000,weights),{'terminal':50.,'win_speed':2.25})
        for outcome in ['lose','draw','unknown']:
            self.assertEqual(terminal_reward_terms(outcome,5100,100,20000,weights)['win_speed'],0.)
        for frame in [None,99,20100]:
            self.assertEqual(terminal_reward_terms('win',frame,100,20000,weights)['win_speed'],0.)

    def test_whole_and_fragment_rewards_values_and_detections_conserve(self):
        cfg=self.cfg(); network=HoK1v1Network(cfg).eval()
        def runner(): return EnvRunner(EventGame(end=5),copy.deepcopy(network),copy.deepcopy(cfg),torch.device('cpu'))
        torch.manual_seed(4); whole=runner().collect_trajectory(matchup=Matchup(112,133,0))
        torch.manual_seed(4); collector=EpisodeCollector(runner(),Matchup(112,133,0),burn_in=4)
        fragments=[collector.collect(4,version=0),collector.collect(4,version=0)]
        self.assertEqual(fragments[0]['sample_stats']['reward_components']['win_speed'],0.)
        stats=fragments[-1]['episode_stats']
        self.assertGreater(stats['reward_components']['win_speed'],0.)
        self.assertEqual(stats['reward_components'],whole['episode_stats']['reward_components'])
        self.assertEqual(stats['reward_detection'],whole['episode_stats']['reward_detection'])
        self.assertAlmostEqual(stats['reward'],sum(stats['reward_components'].values()))
        detection=summarize_reward_detection([stats])
        self.assertEqual(detection['last_hit_coverage'],1.)
        batch=join_trajectories(fragments)
        self.assertLess(PPOLearner(network,cfg)._diagnose(batch,2)['old_new_value_max'],1e-4)

    def test_explicit_reward_migration_and_duplicate_shaping_guards(self):
        original=Config(); original.network.lstm_hidden=16
        source=PPOLearner(HoK1v1Network(original),original)
        cfg=self.cfg(); cfg.network.lstm_time_steps=original.network.lstm_time_steps
        target=PPOLearner(HoK1v1Network(cfg),cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.pt'; source.save(path)
            with self.assertRaisesRegex(ValueError,'reward'): target.initialize_from(path)
            record=target.initialize_from(path,allow_reward_changes=True)
            self.assertTrue(record['reward_changed']); self.assertFalse(target.optimizer.state)
            with self.assertRaises(ValueError): target.load(path)
            migrated=Path(tmp)/'migrated.pt'; target.save(migrated)
            target.load(migrated)
        for change in [{'money':.5},{'hp':1.},{'damage_taken':.5},
                       {'nonhero_damage_taken':.5},{'last_hit':-.1}]:
            changed=copy.deepcopy(cfg)
            for key,value in change.items(): setattr(changed.reward,key,value)
            with self.assertRaises(ValueError): changed.validate()

    def test_nonhero_damage_residual_counts_alive_loss_and_skips_unknown_or_death(self):
        a,b=self.snapshots(); weights=self.cfg().reward
        weights.nonhero_damage_taken=-.75
        b['my'].update(hp=4400.,hurt_by_hero=300.)
        self.assertAlmostEqual(state_reward_terms(a,b,weights)['nonhero_damage_taken'],-.06)
        evidence=step_reward_detection(a,b,2)
        self.assertEqual(evidence['nonhero_damage_taken_raw'],400.)
        self.assertEqual(evidence['nonhero_damage_taken_available_steps'],1)
        b['my']['hp']=0.
        self.assertEqual(state_reward_terms(a,b,weights)['nonhero_damage_taken'],0.)
        self.assertEqual(step_reward_detection(a,b,2)['nonhero_damage_taken_available_steps'],0)
        b['my'].update(hp=4000.,deaths=1.)
        self.assertEqual(state_reward_terms(a,b,weights)['nonhero_damage_taken'],0.)
        b['my'].update(deaths=0.,hurt_by_hero=None)
        self.assertEqual(state_reward_terms(a,b,weights)['nonhero_damage_taken'],0.)
        b['my'].update(hurt_by_hero=900.,hp=4900.)
        self.assertEqual(state_reward_terms(a,b,weights)['nonhero_damage_taken'],0.)


if __name__=='__main__': unittest.main()
