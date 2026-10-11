"""塔前动作保护的安全边界：只收窄有退路的合法移动。"""
import unittest
from types import SimpleNamespace as Obj

import numpy as np

from hok1v1_rl.action_mask import effective_legal_action
from hok1v1_rl.config import ActionConfig


def actor(camp, x, z, *, hp=1000, kind=None, runtime_id=8):
    return Obj(camp=camp, location=Obj(x=x, z=z), hp=hp,
               type=kind, config_id=154, runtime_id=runtime_id)


def state(camp=1, soldier=None, *, distance=5000, has_tower=True):
    tower_x = 11285 if camp == 1 else -11240
    tower_z = 11275 if camp == 1 else -11228
    forward = 1 if camp == 1 else -1
    hero = actor(camp, tower_x-forward*distance, tower_z-forward*distance)
    towers = [actor(3-camp, tower_x, tower_z, hp=5000, kind=21)] if has_tower else []
    legal = np.zeros(172, dtype=np.float32)
    legal[[1, 2, 3]] = 1
    legal[12+1:12+16] = 1
    legal[28+1:28+16] = 1
    return {"legal_action": legal, "player_id": 8,
            "req_pb": Obj(hero_list=[hero], organ_list=towers,
                          soldier_list=[] if soldier is None else [soldier])}


class TowerGuardTest(unittest.TestCase):
    def test_uncovered_tower_blocks_forward_directions_on_both_sides(self):
        for camp in (1, 2):
            with self.subTest(camp=camp):
                current = state(camp)
                guarded = effective_legal_action(current, ActionConfig(rule="targeted_attack_v1",
                                                                        tower_guard_radius=15000))
                for offset in (12, 28):
                    self.assertTrue(np.all(guarded[offset+8:offset+16] == 0))
                    self.assertTrue(np.all(guarded[offset+1:offset+8] == 1))
                self.assertEqual(guarded[2], 1)
                self.assertEqual(current["legal_action"][12+12], 1)

    def test_covering_minion_keeps_sdk_move_options(self):
        current = state(soldier=actor(1, 11000, 11000))
        guarded = effective_legal_action(current, ActionConfig(rule="targeted_attack_v1",
                                                                tower_guard_radius=15000))
        np.testing.assert_array_equal(guarded, current["legal_action"])

    def test_near_tower_boundary_still_blocks_forward_with_minion(self):
        current = state(distance=500, soldier=actor(1, 11200, 11200))
        guarded = effective_legal_action(current, ActionConfig(rule="targeted_attack_v1",
                                                                tower_guard_radius=15000))
        for offset in (12, 28):
            self.assertTrue(np.all(guarded[offset+8:offset+16] == 0))

    def test_outside_radius_and_missing_tower_keep_sdk_options(self):
        config = ActionConfig(rule="targeted_attack_v1", tower_guard_radius=15000)
        for current in (state(distance=15000), state(has_tower=False)):
            np.testing.assert_array_equal(effective_legal_action(current, config), current["legal_action"])

    def test_no_retreat_option_does_not_make_head_empty(self):
        current = state()
        current["legal_action"][12+1:12+8] = 0
        current["legal_action"][28+1:28+8] = 0
        guarded = effective_legal_action(current, ActionConfig(rule="targeted_attack_v1",
                                                                tower_guard_radius=15000))
        np.testing.assert_array_equal(guarded, current["legal_action"])


if __name__ == "__main__":
    unittest.main()
