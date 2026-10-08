"""配置 dataclass 与 YAML 加载器。

────────────────────────────────────────────────────────────────────────────
环境前提（务必先读，否则 A 类问题会反复出现）
────────────────────────────────────────────────────────────────────────────
1. **Python 必须用 3.9**，即 ``conda activate hok``（本项目实测环境）。
   - hok_env 的 ``setup.py`` 写的是 ``python_requires=">=3.6, <3.10"``，
     且原生扩展只提供 3.6/3.7/3.8/3.9 四个 ABI：
         interface.cpython-36m/37m/38/39-x86_64-linux-gnu.so
   - **没有 cpython-310**，所以用 /usr/bin/python3.10 一定 import 失败。
   - 3.8 虽然也能加载 .so，但 **PyTorch cu128 轮子不含 cp38**
     （cu124 最高 2.6.0，cu126/cu128 最低 cp39），
     而 RTX 5070 是 Blackwell sm_120，需要 CUDA 12.8+ 的 torch（≥2.7）。
   - 结论：**3.9 是唯一同时满足"能加载 .so"与"能用 sm_120 GPU"的版本**。
     推荐：``conda create -n hok python=3.9`` +
     ``pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision``

2. 依赖：``pyyaml tensorboard loguru requests h5py pyzmq`` + ``pip install -e .``（hok_env）。

3. GPU 自检（必须包含 sm_120，且真实 kernel 能跑）::

       python -c "import torch; print(torch.cuda.get_arch_list()); \\
                  a=torch.randn(512,512,device='cuda'); print((a@a).sum().item())"

   ⚠️ ``torch.cuda.is_available()`` 返回 True **并不代表能用**：
   在 arch 不匹配时它照样返回 True，但任何 kernel 都会抛
   ``RuntimeError: CUDA error: no kernel image is available for execution on the device``。

────────────────────────────────────────────────────────────────────────────
设计说明：使用 dataclass 而非 dict，便于 IDE 类型提示与字段拼写检查。
``from_yaml`` 采用"默认值 + 覆盖"策略，保证新增字段不会让旧 YAML 加载失败。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import logging
import os

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """硬件约束配置。

    阈值由低到高：warn < degrade < oom，对应不同响应级别。

    注意：RTX 5070 的物理显存为 12227 MiB（≈11.94 GiB），
    下面的 12.0 是标称值；``vram_oom_gb`` 不宜贴得太近，
    因为 PyTorch 缓存分配器会保留（reserved）大量显存，
    真正的 OOM 往往发生在 allocated 远低于物理上限时。
    """
    vram_total_gb: float = 12.0
    vram_warn_gb: float = 9.0
    vram_degrade_gb: float = 10.5   # 触发降级：先降 num_envs，再降 seq_len，再降 attn_heads
    vram_oom_gb: float = 11.5       # 临界 OOM，需立即释放显存
    ram_total_gb: float = 32.0


@dataclass
class TrainConfig:
    """训练通用配置。"""
    backend: str = "pytorch"        # 仅支持 PyTorch，TF 后端为 TODO
    # ⚠️ 默认关闭 AMP。理由：本网络约 5M 参数、minibatch 仅 1024 帧
    # （激活 ~百 MB 量级），12G 显存根本用不到混合精度；而 AMP 会让
    # **采样侧（fp32，torch.no_grad 且无 autocast）与训练侧（fp16）**
    # 对同一帧算出略有差异的 logits，于是 PPO 的 old_log_probs 与
    # new_log_probs 在权重未更新时 ratio 也 != 1，KL 诊断被污染。
    # 关掉它可以让"第一个 minibatch 的 ratio 应恒等于 1"成为可用的自检。
    amp: bool = False
    grad_clip: float = 0.5          # 与 PPOConfig.max_grad_norm 保持一致（二者同义）
    seed: int = 42                  # 全局随机种子，由 train_ppo 在启动时真正设置
    log_dir: str = "./logs"
    tb_log_dir: str = "./logs/tb"
    wandb_project: str = "hok1v1_rl"
    wandb_enabled: bool = False
    log_interval_steps: int = 50
    save_interval_steps: int = 200      # 每多少步保存一次 checkpoint（ckpt_step_{n}.pt）
    eval_interval_steps: int = 200      # 每多少步做一次确定性评估（不打入训练 batch）
    eval_episodes: int = 2              # 每次评估的对局数
    dashboard_port: int = 8600          # scripts/dashboard.py 默认监听端口


@dataclass
class EnvConfig:
    """Gamecore 环境连接与并行配置。

    ⚠️ 字段语义已修正以匹配官方 ``HoK1v1`` 的构造签名
    （hok_env/hok/hok1v1/env1v1.py:29-38）::

        HoK1v1(runtime_id, game_launcher, lib_processor, addrs,
               eval_mode=False, predict_frequency=3, aiserver_ip="127.0.0.1")

    其中 ``addrs`` 是**每个玩家一个**的 zmq 地址列表（1v1 即 2 个），
    不是"每个环境一个端口"。原 ``port_begin`` 的语义与之不符，已替换为
    ``zmq_port_begin``（起始端口，player i 用 zmq_port_begin + i）。
    """
    num_envs: int = 1                 # 并行环境数。当前 EnvRunner 只实现单环境；
                                      # 多环境需 fork 子进程，见 env_runner 顶部说明。
    player_num: int = 2               # HoK1v1.PLAYER_NUM，1v1 固定为 2
    # 运行时标识，gamecore 侧拼进 game_id（kaiwu- 前缀）。模拟器不允许含 '_'，
    # 否则 Windows 侧模拟器直接退出，对局无法开始（build_hok_env 内有校验）
    runtime_id: str = "hok1v1rl-train"
    gamecore_server_addr: str = "127.0.0.1:23432"   # Windows 侧 gamecore-server 监听地址
    # gamecore（Windows 侧）连回本机 zmq 的 IP，zmq 也绑定在该 IP 上。
    # 127.0.0.1 在 WSL2 两种网络模式下都可用（NAT 走 localhost 转发，
    # 镜像模式走共享回环）。gamecore 跑在远端机器时改为本机局域网 IP。
    # 空串 = 自动探测：镜像模式下会探测到 Windows 自己的适配器 IP，
    # Windows 连回该 IP 会 SYN 挂起，对局无法开始，不推荐。
    ai_server_addr: str = "127.0.0.1"
    zmq_port_begin: int = 35150       # addrs 起始端口：player i -> tcp://{ai_server_addr}:{begin+i}
    gamecore_req_timeout: int = 3000  # GamecoreClient 请求超时（毫秒）
    max_frame_num: int = 20000        # 单局最大帧数，超过则强制结束，防止卡死局
    predict_frequency: int = 3        # 每 3 帧推理一次（由 env 内部按帧号取模实现）
    # 对手使用内置 common_ai（True）而不是让我们的网络接管第二方。
    # 1v1 单智能体训练时必须为 True，否则 step() 需要同时给出两个玩家的动作。
    use_common_ai_opponent: bool = True
    # 时间惩罚：每步常数 -coef。
    # ⚠️ 在**已改回官方量级的奖励**（推塔 5.0/点、换血 2.0/点）下，
    # 0.001/步 累计约 -6.7/局，相对一局 ~10^5 的 dense 回报可以忽略，
    # 也就是说它现在实际上不起作用。这是刻意保留的保守值：
    # "拖时间刷收益"的问题已经由两件事结构性解决 ——
    #   (1) money/exp 改成零和差值且权重回到官方量级（被动收入不再给钱）；
    #   (2) 达到 max_frame_num 的截断会吃 RewardWeights.truncate 惩罚。
    # 若仍希望显式加快终结节奏，可调到 0.1~1.0（一局累计 666~6667）。
    reward_time_coef: float = 0.001


@dataclass
class NetworkConfig:
    """网络结构配置：LSTM + Self-Attention + Multi-head Value。

    对应官方 baseline ``aiarena/1v1/common/algorithm_torch.py``：
    - ``lstm_hidden=512``、``lstm_time_steps=16``、``target_embed_dim=32`` 与官方一致
    - ``attn_heads``、``value_heads`` 为本项目的增强项（官方没有这两块）
    """
    # ⚠️ 725 = 705(gamecore 原始观测) + 20(英雄身份 one-hot)。
    # 官方 aiarena/1v1/actor/custom.py 会把我方英雄 id 转成 20 维 one-hot
    # 追加到观测末尾，网络里的 global 段因此是 25 维（5 game_time + 20 英雄身份）。
    # 若按 705 建网，策略就不知道自己在玩哪个英雄，而本项目每局轮换英雄 ——
    # 详见 network.py 顶部的布局说明。
    feature_dim: int = 725
    #: 是否在观测末尾追加 20 维英雄身份 one-hot（复刻官方 custom.py）。
    #: 置 False 会退回原始 705 维，此时必须同步把 feature_dim 改成 705，
    #: 否则 FeatureEncoder 的维度断言会直接报错（这是刻意的，避免静默错位）。
    use_hero_identity: bool = True
    legal_action_dim: int = 84      # 训练样本中的压缩 legal_action 维度（官方 SERI_VEC_SPLIT_SHAPE[1]）
    lstm_hidden: int = 512
    lstm_time_steps: int = 16       # 16 帧滑窗（BPTT 截断长度），覆盖短期战术记忆
    attn_heads: int = 4
    attn_d_ff: int = 1024           # Feed-forward 中间维度，通常为 hidden 的 2 倍
    # ⚠️ 必须为 0。采样侧 network.eval() 关闭 dropout，训练侧 network.train()
    # 打开 dropout，两侧前向因此不是同一个函数：old/new log prob 的差值被
    # dropout 噪声污染，而 KL 估计量 (exp(d)-1-d) 恒 >= 0，
    # 零均值噪声也会**单调抬高**测得的 KL，直接触发 kl_target/kl_max 早停
    # （症状：minibatch 只用 1/8、策略几乎不更新）。
    attn_dropout: float = 0.0
    # ⚠️ 时间维 self-attention 默认关闭，见 network.MultiHeadSelfAttention
    # 的 docstring：采样侧每次只喂 T=1 的单帧，attention 退化为恒等映射，
    # 而训练侧喂 T=16 的 chunk，两侧前向不一致 => PPO ratio/KL 全部失真。
    use_time_attention: bool = False
    # 6 头分层动作空间：button(12) + move_x/z(16) + kill_x/z(16) + target(8)
    label_size_list: List[int] = field(
        default_factory=lambda: [12, 16, 16, 16, 16, 8]
    )
    # Multi-head Value：把 value 分解为 6 个子目标，加速信用分配（本项目增强项）
    value_heads: List[str] = field(
        default_factory=lambda: ["hp", "money", "tower", "kill", "dead", "win"]
    )
    target_embed_dim: int = 32      # target 头 embedding 维度，与官方 TARGET_EMBED_DIM 一致


@dataclass
class PPOConfig:
    """PPO 超参。

    取值理由：
    - ``gamma=0.995`` / ``lamda=0.95``：官方 GAMMA / LAMDA
    - ``clip_param=0.2``：官方 CLIP_PARAM；``ratio_clamp_high=3.0`` 对应官方 ratio.clamp(0,3)
    - ``entropy_beta=0.01``（**实测调参，非官方值**）：60 步实测 entropy 钉在
      1.92~2.01 而 kl 只有 1e-4 → 熵项压过 advantage 信号，先降到 0.01
    - ``lr=3e-4``（**实测调参，非官方值**）/ ``lr_end`` 同值：**不退火**。
      官方 1e-4 配的是 4 倍大的 batch；本项目实测 kl 太小、策略几乎没被推动
    - ``ppo_epochs=2`` + ``minibatch_size == batch_size``（**实测调参**）：
      整批做 2 次 SGD。官方是 1 次，但其 batch 有 512 条序列；本项目 batch
      只有 4~5 局且高度相关，多 1 个 epoch 用于提高样本复用
    - ``normalize_returns=False``：**关闭**在线归一化（官方全程没有归一化）。
      开启时 explained_variance 的分子分母基准不一致，该指标不可读
    - ``kl_max``：KL 硬阈值，超过立即停止当前 epoch
    - ``kl_target``：KL 软阈值，超过则跳过本轮剩余 minibatch（"双阈值"中的软的那个）
      ⚠️ 官方**没有** KL 早停；样本量小时它会随机砍掉梯度步。
    """
    gamma: float = 0.995
    lamda: float = 0.95
    clip_param: float = 0.2
    ratio_clamp_low: float = 0.0      # 官方: ratio.clamp(0.0, 3.0)
    ratio_clamp_high: float = 3.0
    #: ⚠️ 0.01（**不是**官方默认的 0.025）。
    #: 实测依据：60 步长跑中 `train/entropy` 长期钉在 1.92~2.01（接近混合动作分布的
    #: 上限），而 `train/kl` 只有 1e-4 量级 —— 说明熵项压过了 advantage 信号，
    #: 策略几乎没被推动。官方 0.025 是在别的配置下调出来的，这里先降到 0.01。
    #: 若 entropy 崩到 <0.3 再回调。
    entropy_beta_start: float = 0.01
    entropy_beta_end: float = 0.01     # = start，即不退火
    value_coef: float = 0.5
    #: ⚠️ 默认关闭。官方 value loss 就是 `0.5*mean((return - V)^2)`，**没有**
    #: value clipping。开启它需要 `old_values` 与新值同尺度，而 old_values
    #: 来自采样侧、新值来自训练侧：一旦两侧前向有任何不一致（见
    #: MultiHeadSelfAttention 的说明），v_clipped 就没有意义。
    #: 先把这一项关掉与官方对齐，确认训练正常后再单独做对比实验。
    #: ⚠️ 关闭（对齐官方）。官方 value loss 就是 `0.5*mean((return-V)^2)`，
    #: **没有** value clipping。开启它需要 old_values 与新值同尺度，而
    #: old_values 来自采样侧、新值来自训练侧：一旦两侧前向有任何不一致，
    #: v_clipped 就失去意义（见 MultiHeadSelfAttention 的说明）。
    value_clip: bool = False
    #: ⚠️ **已关闭**。官方全程没有 return/advantage 归一化，而本实现用的是
    #: `RunningRewardNormalizer`（**全局运行** mean/var，见 reward.py）。
    #: 它有两个副作用：
    #:   ① 同一个 return 在不同 batch 里回归目标不同（早期统计量还在漂移）；
    #:   ② `explained_variance` 的分子用了去均值的 returns、分母却用原始
    #:      尺度，两者基准不一致 → 该指标失去意义（实测长期贴 0、个别步 −146）。
    #: 关掉后 value 直接拟合原始 λ-return，`explained_variance` 才可读。
    normalize_returns: bool = False
    #: ⚠️ 3e-4（**不是**官方的 1e-4）。依据：60 步实测 `train/kl` 只有 1e-4 量级、
    #: `clip_fraction` 0.01%（健康 PPO 的 KL 应在 5e-3~2e-2），即策略几乎没被推动。
    #: 官方 1e-4 是在 batch 大 4 倍的配置下调出来的，这里先升到 3e-4。
    #: 若 `train/kl` 冲上 0.02 或 `early_stopped` 频繁，就回调。
    lr: float = 3.0e-4
    #: ⚠️ 不退火（对齐官方）。官方把 INIT_LEARNING_RATE_START 赋给
    #: `self.learning_rate` 后再未修改，且 Algorithm 没有 get_lr_scheduler。
    #: 旧默认 0.0 会在 total_steps 内把 LR 线性退到 0 —— 配合"短测把
    #: --total-steps 调小"的用法等于学习率瞬间归零（见诊断文档 P1）。
    #: 这里保持 lr_end == lr，即退火系数恒为 1（见 learner.set_lr_schedule）。
    lr_end: float = 3.0e-4            # = lr，即不退火
    #: ⚠️ 2（**不是**官方的 1）。官方是 batch=512 序列 + 1 次 SGD，而本项目
    #: batch 只有 4~5 局（~8192 决策）、且连续对局高度相关 —— 有效样本量约为
    #: 官方的 1/4。同一批数据多跑 1 个 epoch（2 次 SGD）是提高样本复用、
    #: 降低 advantage 方差的常规做法。
    #: （训练路径仍不受影响：minibatch_size == batch_size，所以是"整批 × 2 次"。）
    ppo_epochs: int = 2
    minibatch_size: int = 8192        # 以"帧"为单位；= batch_size 时整批一次更新
    batch_size: int = 8192            # 单次 update 收集的决策数（= 官方 512×16）
    kl_target: float = 0.015          # 软阈值
    kl_max: float = 0.05              # 硬阈值
    max_grad_norm: float = 0.5


@dataclass
class BCConfig:
    """Behavior Cloning 阶段配置（尚未实现，保留占位）。"""
    enabled: bool = False
    lr: float = 1.0e-3
    alpha_start: float = 1.0
    alpha_end: float = 0.0
    alpha_decay_steps: int = 10_000_000


@dataclass
class SelfPlayConfig:
    """Self-Play 模型池配置（尚未实现，保留占位）。"""
    enabled: bool = False
    model_pool_dir: str = "./model_pool"
    latest_ratio: float = 0.8
    history_ratio: float = 0.2
    snapshot_interval: int = 100
    max_snapshots: int = 20
    elo_k: int = 32


@dataclass
class Config:
    """顶层配置容器，组合所有子配置。"""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置。

        加载策略：先实例化默认配置，再用 YAML 中的字段覆盖，
        这样未在 YAML 中声明的字段仍使用默认值，避免新增字段时旧 YAML 加载失败。

        ⚠️ 与原实现的区别：**未知字段会打 warning**。
        原实现用 ``hasattr`` 静默丢弃，把 ``minibatch_size`` 误拼成
        ``minibatchsize`` 时完全不报错，是很容易踩的坑。

        Args:
            path: YAML 路径
        Returns:
            Config 实例
        """
        # encoding="utf-8" 是必需的：default.yaml 含中文注释，
        # 在 Windows（locale=cp936）下用默认编码打开会 UnicodeDecodeError。
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        for section_name, section_data in data.items():
            if not hasattr(cfg, section_name):
                logger.warning("配置节 '%s' 在 Config 中不存在，已忽略", section_name)
                continue
            if not isinstance(section_data, dict):
                logger.warning("配置节 '%s' 不是字典，已忽略", section_name)
                continue
            section = getattr(cfg, section_name)
            for k, v in section_data.items():
                if not hasattr(section, k):
                    logger.warning("未知配置项 '%s.%s'（疑似拼写错误），已忽略",
                                   section_name, k)
                    continue
                setattr(section, k, v)
        return cfg

    @classmethod
    def default(cls) -> "Config":
        """返回默认配置（不读 YAML）。"""
        return cls()

    def to_dict(self) -> dict:
        """转 dict，便于序列化或日志记录。"""
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """加载配置：path 为空或文件不存在则返回默认配置。"""
    if path and os.path.exists(path):
        return Config.from_yaml(path)
    return Config.default()
