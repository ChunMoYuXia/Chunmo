"""日志模块：统一日志输出到控制台与文件。

设计要点：
    - 采用「懒初始化 + 全局单例」模式：首次调用 get_logger 时才配置 handler，
      避免 import 阶段就创建日志目录/文件，减少副作用。
    - 控制台只输出 INFO 及以上级别（面向用户），文件记录 DEBUG 及以上
      （面向开发者排查问题），两者级别分离。
    - 使用 RotatingFileHandler 限制单文件 2MB、保留 3 份备份，
      防止日志无限增长撑爆磁盘。
    - 日志目录不可写时降级为仅控制台输出，保证程序不因日志故障崩溃。
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from .paths import LOG_DIR, ensure_dirs

# 根 logger 名称，所有子 logger（如 "quantai.audit"）都挂在该名称下
_LOGGER_NAME = "quantai"
# 全局初始化标记，确保 _init_logger 只执行一次
_initialized = False


def get_logger(name: str | None = None) -> logging.Logger:
    """获取配置好的 logger。

    Args:
        name: 子 logger 名称（如 "audit"），传 None 则返回根 logger。
              传 "audit" 实际得到 "quantai.audit"，自动继承根 logger 的 handler。

    Returns:
        logging.Logger: 已配置 handler 的 logger 实例。

    线程安全说明：_initialized 检查在 CPython GIL 下是原子的，
    多线程首次并发调用时可能多次进入 _init_logger，
    但 _init_logger 内部有 handlers 非空判断做二次守卫，不会重复添加。
    """
    global _initialized
    if not _initialized:
        _init_logger()
        _initialized = True
    return logging.getLogger(name) if name else logging.getLogger(_LOGGER_NAME)


def _init_logger() -> None:
    """配置根 logger 的 handler 与格式。

    只应被 get_logger 调用一次。重复调用由 handlers 非空判断拦截，
    防止同一 handler 被多次添加导致日志重复输出。
    """
    ensure_dirs()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # 关闭 propagate，避免日志同时被 Python 根 logger 处理而输出两遍
    logger.propagate = False

    # 二次守卫：如果已有 handler 说明已初始化，直接返回
    if logger.handlers:
        return

    # 统一日志格式：时间 [级别] logger名: 消息
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler：INFO 级别，输出到 stdout 而非 stderr，
    # 便于在网页端/终端统一捕获
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        # 文件 handler：DEBUG 级别，记录更详细的调试信息
        # 2MB 单文件 + 3 份备份，最多占用约 8MB 磁盘空间
        fh = RotatingFileHandler(
            LOG_DIR / "quantai.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        # 日志目录不可写时降级为仅控制台
        pass
