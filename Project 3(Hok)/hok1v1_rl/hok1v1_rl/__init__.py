"""HoK 1v1 RL 训练包 —— PPO + LSTM/Attention + Multi-head Value 实现。

模块职责：
  * ``network.py``      网络结构（LSTM + 4-head Self-Attention + 6-head Value）
  * ``learner.py``      PPO 训练循环（逐头 ratio / GAE / KL 双阈值 / AMP / GradClip）
  * ``env_runner.py``   HoK1v1 环境封装、分层动作采样、轨迹收集
  * ``action_mask.py``  分层动作空间的 mask 工具与 172→84 legal_action 压缩
  * ``reward.py``       dense reward 解析与 GAE
  * ``vram_guard.py``   显存监控与降级回调
  * ``logger.py``       TensorBoard / WandB 统一日志

────────────────────────────────────────────────────────────────────────────
运行前提（务必先读 config.py 顶部的环境说明）
────────────────────────────────────────────────────────────────────────────
  * Python 3.9（hok_env 原生扩展只支持 3.6–3.9；3.8 用不了 cu128 的 torch）
  * torch >= 2.7 + CUDA 12.8 轮子（RTX 5070 是 Blackwell sm_120）
  * gamecore-server 在 Windows 侧运行并监听 23432

快速自检::

    conda activate hok
    python scripts/_smoke_test.py     # 纯离线，不需要 gamecore
    python scripts/test_env.py        # 需要 gamecore 已启动
"""
__version__ = "0.2.0"
