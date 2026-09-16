"""操作审计日志：记录用户操作，便于追溯纠错。

日志格式为 JSON Lines（每行一条 JSON），存于 logs/audit.jsonl。
提供记录、查询、清理接口，网页可直接查看。

选型理由：
    - 选用 JSON Lines 而非普通文本日志，因为每条记录结构固定，
      便于后续用 jq/pandas 做统计分析，也方便网页端直接解析展示。
    - 追加写入（"a" 模式）保证并发安全语义：每条记录原子写入一行，
      即使进程异常退出，已写入的行仍然完整可读。
    - 不使用数据库，是因为审计日志量小、查询简单，文件方案零依赖、
      易备份、可直接 grep。
"""
import json
import time
from pathlib import Path

from .paths import LOG_DIR, ensure_dirs
from .logger import get_logger

log = get_logger("audit")
AUDIT_FILE = LOG_DIR / "audit.jsonl"


def _ensure() -> None:
    """确保日志目录存在。

    作为所有读写操作的前置守卫，避免首次运行时因目录缺失导致 OSError。
    抽成独立函数而非在每个函数内重复调用，保持 DRY。
    """
    ensure_dirs()


def record(action: str, user_input: dict | None = None,
           result: str = "success", error: str | None = None,
           duration_ms: int | None = None, **extra) -> None:
    """记录一条操作审计日志。

    Args:
        action: 操作类型（生成研报/分组对比/多模型预测/修改设置/导出/错误）
        user_input: 用户输入参数（股票代码、日期等），为 None 时不写入
        result: success / failed，默认 success
        error: 错误信息（如有），为空时不写入
        duration_ms: 耗时毫秒，为 None 时不写入
        **extra: 额外字段，会合并到日志条目末尾，用于扩展（如 model 名称）

    容错策略：
        - 写入失败时仅通过 logger 警告，不抛出异常。
          审计日志是辅助功能，绝不能因日志写入失败影响主业务流程。
    """
    _ensure()
    entry = {
        # 同时存可读时间戳和 epoch 秒，前者便于人眼阅读，后者便于排序计算
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ts_epoch": int(time.time()),
        "action": action,
        "result": result,
    }
    if user_input:
        entry["input"] = user_input
    if error:
        entry["error"] = error
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    entry.update(extra)
    try:
        # 追加模式 + 单次 write，保证单条记录不会被截断
        with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("审计日志写入失败：%s", exc)


def query(limit: int = 100, action: str | None = None,
          result: str | None = None) -> list[dict]:
    """查询审计日志（最新在前）。

    Args:
        limit: 返回条数，默认 100，防止日志过大时一次性加载过多
        action: 按操作类型过滤，None 表示不过滤
        result: 按结果过滤（success/failed），None 表示不过滤

    Returns:
        list[dict]: 日志条目列表，按时间倒序（最新在前）

    容错策略：
        - 文件不存在时返回空列表。
        - 逐行解析时遇到损坏行（JSONDecodeError）直接跳过，
          保证单行损坏不影响整个查询结果。
    """
    _ensure()
    if not AUDIT_FILE.exists():
        return []
    entries = []
    with open(AUDIT_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过损坏行，不中断整个读取流程
                continue
    # 过滤
    if action:
        entries = [e for e in entries if e.get("action") == action]
    if result:
        entries = [e for e in entries if e.get("result") == result]
    entries.reverse()  # 最新在前
    return entries[:limit]


def stats() -> dict:
    """返回日志统计：总条数、各 action 计数、失败数。

    Returns:
        dict: {"total": int, "by_action": {action: count}, "failed": int}

    用途：网页端仪表盘展示操作概览，无需加载全部日志内容。
    """
    _ensure()
    if not AUDIT_FILE.exists():
        return {"total": 0, "by_action": {}, "failed": 0}
    total = 0
    by_action: dict[str, int] = {}
    failed = 0
    with open(AUDIT_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            act = e.get("action", "unknown")
            by_action[act] = by_action.get(act, 0) + 1
            if e.get("result") == "failed":
                failed += 1
    return {"total": total, "by_action": by_action, "failed": failed}


def clear() -> int:
    """清空审计日志，返回删除的条数。

    Returns:
        int: 被删除的日志行数（用于前端确认提示）。

    实现说明：直接 unlink 文件而非 truncate，是因为下次 record 时
    会自动重新创建，逻辑更简单。
    """
    _ensure()
    if not AUDIT_FILE.exists():
        return 0
    # 先统计行数再删除，用于向用户反馈清空了多少条
    count = sum(1 for _ in open(AUDIT_FILE, encoding="utf-8"))
    AUDIT_FILE.unlink()
    return count
