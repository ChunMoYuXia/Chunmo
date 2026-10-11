"""配置入口：dataclass 提供类型提示，YAML 覆盖默认值，未知字段立即报错。"""
from dataclasses import asdict, dataclass, field, fields
import math
from typing import List, Optional
import yaml

HERO_IDS = [112,121,123,131,132,133,140,141,146,150,154,157,163,169,175,182,193,199,502,513]
# v2 保证轨迹记录模型的真实价值预测，不被行为统计覆盖。
ENVIRONMENT_CONTRACT = "fresh-remote-control-value-v2"

@dataclass
class HardwareConfig:
    vram_total_gb: float = 12.0
    vram_warn_gb: float = 9.0
    vram_degrade_gb: float = 10.5
    vram_oom_gb: float = 11.5
    ram_total_gb: float = 32.0

@dataclass
class TrainConfig:
    seed: int = 42
    log_dir: str = "./runs"
    tb_log_dir: str = "./runs/tb"
    wandb_project: str = "hok1v1_rl"
    wandb_enabled: bool = False
    log_interval_steps: int = 1
    save_interval_steps: int = 20
    eval_interval_steps: int = 50
    eval_episodes: int = 40  # 快评：20 英雄 × 双方，仅供观察
    dashboard_port: int = 8600

@dataclass
class EnvConfig:
    num_envs: int = 1
    player_num: int = 2
    runtime_id: str = "hok1v1rl-train"
    gamecore_server_addr: str = "127.0.0.1:23432"
    ai_server_addr: str = "127.0.0.1"
    zmq_port_begin: int = 35150
    gamecore_req_timeout: int = 3000
    max_frame_num: int = 20000
    predict_frequency: int = 3
    use_common_ai_opponent: bool = True
    reward_time_coef: float = 0.0
    hero_ids: List[int] = field(default_factory=lambda: HERO_IDS.copy())
    opponent_hero_ids: List[int] = field(default_factory=lambda: HERO_IDS.copy())
    max_decisions: int = 0  # 0=自然结束；非零只用于快速调试

@dataclass
class NetworkConfig:
    feature_dim: int = 725
    use_hero_identity: bool = True
    legal_action_dim: int = 84
    lstm_hidden: int = 512
    lstm_time_steps: int = 16
    label_size_list: List[int] = field(default_factory=lambda: [12,16,16,16,16,8])
    target_embed_dim: int = 32

@dataclass
class PPOConfig:
    gamma: float = 0.999  # 决策步折扣，延长长局的回报视野
    lamda: float = 0.95
    clip_param: float = 0.2
    entropy_beta_start: float = 0.01
    entropy_beta_end: float = 0.01
    value_coef: float = 0.5
    value_clip: bool = False
    lr: float = 0.0001
    lr_end: float = 0.0001
    ppo_epochs: int = 2
    minibatch_size: int = 2048
    batch_size: int = 8192
    kl_target: float = 0.015
    max_grad_norm: float = 0.5
    grad_clip_mode: str = "global"  # 原基线；separate 分别裁剪共享层、动作头、价值头

@dataclass
class RewardConfig:
    money: float = 0.5
    exp: float = 0.0  # 经验是否累计尚未确认，先关掉升级回绕的潜在误奖励
    hp: float = 1.0
    ep: float = 0.0
    kill: float = 2.0
    dead: float = -2.0
    tower: float = 10.0
    last_hit: float = 0.0
    win: float = 50.0
    lose: float = -50.0
    economy_diff: float = 0.0  # 双方累计收入差的变化 / 1000
    damage_dealt: float = 0.0  # 对英雄伤害 / 敌方最大生命
    damage_taken: float = 0.0  # 受到英雄伤害 / 我方最大生命，系数为负
    nonhero_damage_taken: float = 0.0  # 存活期间非英雄来源血量损失残差 / 我方最大生命
    win_speed: float = 0.0  # 仅获胜时，按剩余游戏帧比例发放

@dataclass
class EvaluationConfig:
    seed: int = 20261010
    goal_win_rate: float = 0.90
    final_episodes: int = 800  # 20 × 20 × 双方，全配对评估

@dataclass
class ActionConfig:
    # 领域约束实验，默认保持 SDK 行为；不应用于关羽等特殊普攻英雄。
    rule: str = "sdk"
    targeted_attack_hero_ids: List[int] = field(default_factory=lambda: [112, 133, 169])
    # 0 表示关闭；仅在敌方一塔仍存活、附近无己方兵线时限制继续向前移动。
    tower_guard_radius: float = 0.0

@dataclass
class Config:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    @classmethod
    def from_dict(cls, data):
        cfg = cls()
        for name, values in data.items():
            if name not in {f.name for f in fields(cfg)} or not isinstance(values, dict):
                raise ValueError(f"未知配置节或格式错误：{name}")
            section = getattr(cfg, name)
            known = {f.name for f in fields(section)}
            for key, value in values.items():
                if key not in known:
                    raise ValueError(f"未知配置项：{name}.{key}，请检查拼写/旧配置")
                setattr(section, key, value)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    @classmethod
    def default(cls):
        return cls()

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """不支持的选项启动即报错，避免配置看起来生效、实际却被忽略。"""
        for section, values in self.to_dict().items():
            for name, value in values.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"配置 {section}.{name} 必须为有限数")
        if self.env.num_envs != 1 or self.env.player_num != 2:
            raise ValueError("每个进程只支持一个 1v1 环境；多进程采样请使用 --samplers")
        if not self.env.use_common_ai_opponent:
            raise ValueError("当前对手必须为 common AI；尚未实现自博弈")
        r = self.reward
        if r.ep != 0:
            raise ValueError("ep 暂无可靠状态奖励实现，请保持为 0")
        if (min(r.economy_diff, r.damage_dealt, r.last_hit, r.win_speed) < 0 or
                max(r.damage_taken, r.nonhero_damage_taken) > 0):
            raise ValueError("经济差/造成伤害/补刀/获胜提速系数不能为负，受伤系数不能为正")
        if r.money != 0 and r.economy_diff != 0:
            raise ValueError("money 与 economy_diff 不能同时启用，避免重复奖励经济")
        if r.hp != 0 and (r.damage_dealt != 0 or r.damage_taken != 0):
            raise ValueError("hp 与拆分 damage 奖励不能同时启用，避免重复奖励伤害")
        for pool in (self.env.hero_ids, self.env.opponent_hero_ids):
            if not pool or len(pool) != len(set(pool)) or not set(pool) <= set(HERO_IDS):
                raise ValueError("英雄池应为不重复的受支持英雄 ID，见 HERO_IDS")
        n = self.network
        if n.feature_dim != 725 or not n.use_hero_identity:
            raise ValueError("当前使用 705 原始观测 + 20 英雄身份的固定布局")
        if n.target_embed_dim != 32 or n.label_size_list != [12,16,16,16,16,8] or n.legal_action_dim != 84:
            raise ValueError("动作空间和目标嵌入尺寸必须与 SDK 一致")
        positive = [n.lstm_hidden, n.lstm_time_steps, self.ppo.batch_size,
                    self.ppo.minibatch_size, self.ppo.ppo_epochs, self.ppo.lr,
                    self.ppo.kl_target, self.ppo.max_grad_norm, self.env.max_frame_num,
                    self.env.predict_frequency, self.train.save_interval_steps]
        if any(x <= 0 for x in positive) or self.env.max_decisions < 0:
            raise ValueError("网络、采样和 PPO 参数必须为正，max_decisions >= 0")
        if not (0 < self.ppo.gamma <= 1 and 0 <= self.ppo.lamda <= 1):
            raise ValueError("gamma 应在 (0,1]，lamda 应在 [0,1]")
        if self.ppo.grad_clip_mode not in ("global", "separate"):
            raise ValueError("grad_clip_mode 只能为 global 或 separate")
        if self.action.rule not in ("sdk", "targeted_attack_v1"):
            raise ValueError("action.rule 只能为 sdk 或 targeted_attack_v1")
        heroes = self.action.targeted_attack_hero_ids
        if not heroes or len(heroes) != len(set(heroes)) or not set(heroes) <= {112, 133, 169}:
            raise ValueError("普攻约束目前只支持候选英雄 112、133、169；特殊英雄须另行验证")
        if not math.isfinite(self.action.tower_guard_radius) or not 0 <= self.action.tower_guard_radius <= 30000:
            raise ValueError("tower_guard_radius 应在 0..30000，0 表示关闭")
        if (not 0 < self.ppo.clip_param < 1 or self.ppo.lr_end < 0 or
                min(self.ppo.value_coef, self.ppo.entropy_beta_start, self.ppo.entropy_beta_end) < 0):
            raise ValueError("clip_param 应在 (0,1)，学习率终值、value/entropy 系数不能为负")
        if not self.hardware.vram_warn_gb < self.hardware.vram_degrade_gb < self.hardware.vram_oom_gb:
            raise ValueError("显存阈值应满足 warn < degrade < oom")
        if self.train.eval_interval_steps < 0 or self.train.eval_episodes <= 0:
            raise ValueError("eval_interval_steps >= 0，eval_episodes > 0")
        if not 0 < self.evaluation.goal_win_rate < 1 or self.evaluation.final_episodes <= 0:
            raise ValueError("评估目标需在 (0,1)，评估局数必须为正")
        if "_" in self.env.runtime_id or not 1024 <= self.env.zmq_port_begin <= 65534:
            raise ValueError("runtime_id 不能含下划线；ZMQ 起始端口范围 1024..65534")


def load_config(path: Optional[str] = None) -> Config:
    # 显式传入的错误路径应报错，不能偷偷退回默认配置。
    return Config.from_yaml(path) if path else Config.default()
