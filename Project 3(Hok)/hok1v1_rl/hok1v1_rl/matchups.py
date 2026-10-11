"""阵容安排使用独立 RNG；评估不会改变训练的英雄顺序。"""
from dataclasses import dataclass, asdict
import random

@dataclass(frozen=True)
class Matchup:
    hero_id: int
    opponent_id: int
    player_id: int

    def camp_config(self):
        heroes = [None, None]
        heroes[self.player_id] = [{"hero_id": self.hero_id}]
        heroes[1 - self.player_id] = [{"hero_id": self.opponent_id}]
        return {"mode": "1v1", "heroes": heroes}

    def to_dict(self):
        return asdict(self)

class TrainingSchedule:
    """每轮让每个己方英雄在双方各出现一次，对手随机，避免小样本偏科。"""
    def __init__(self, heroes, opponents, seed):
        self.heroes, self.opponents = list(heroes), list(opponents)
        self.rng = random.Random(seed)
        self.pending = []

    def next(self):
        if not self.pending:
            self.pending = [Matchup(hero, self.rng.choice(self.opponents), side)
                            for hero in self.heroes for side in (0, 1)]
            self.rng.shuffle(self.pending)
        return self.pending.pop()

    def state_dict(self):
        return {"rng": self.rng.getstate(), "pending": self.pending.copy()}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.pending = state["pending"].copy()


def evaluation_schedule(heroes, opponents, seed, episodes):
    """固定阵容表；每个完整 2×英雄数区块覆盖全部己方英雄和双方。

    第一个全配对周期内不重复英雄配对。相同 seed 可重复阵容顺序，
    gamecore 本身没有在这里被设置随机种子，故不声称战斗完全确定。
    """
    rng = random.Random(seed)
    opponents_for = {hero: rng.sample(list(opponents), len(opponents)) for hero in heroes}
    result = []
    while len(result) < episodes:
        for round_index in range(len(opponents)):
            block = [Matchup(hero, opponents_for[hero][round_index], side)
                     for hero in heroes for side in (0, 1)]
            rng.shuffle(block)
            result.extend(block)
    return result[:episodes]
