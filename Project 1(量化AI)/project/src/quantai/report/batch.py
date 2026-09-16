"""批量管道：清单里每只股票跑一遍单股全流程；单只失败跳过留档。

本模块是 run_one 的批量封装，解决「一次跑几十上百只股票」时的两个痛点：
1. 单只失败不能拖垮整个批次 → 每只独立 try/except，失败留档继续。
2. 长任务需要进度反馈 → 提供可选的 progress_cb 回调，支持 CLI 进度条 / Web 推送。
"""
from .generator import run_one


def run_batch(codes, start, end, style, provider, refresh=False, progress_cb=None):
    """对清单循环；每只一层 try/except。progress_cb(i, total, code) 可选回调。

    参数：
        codes:       股票代码列表（或可迭代对象）。
        start/end:   区间日期，透传给 run_one。
        style:       AI 股评风格，透传给 run_one。
        provider:    LLM 提供商标识。
        refresh:     是否强制刷新缓存。
        progress_cb: 可选回调函数，签名 progress_cb(i, total, code, ok)：
                     - i: 当前已处理数量（从 1 开始）
                     - total: 总数量
                     - code: 当前股票代码
                     - ok: 本次是否成功（bool）
                     传入 None 表示不回调，适用于无界面场景。

    返回：
        list[dict]: 每个元素结构为 {
            "code": str, "ok": bool,
            "fig": Figure|None, "report": str|None, "df": DataFrame|None,
            "error": str|None
        }。失败时 fig/report/df 为 None，error 为异常字符串。

    容错策略：
        - 捕获所有 Exception，转成字符串存入 error 字段，保证批次不会因
          网络抖动 / 单只股票停牌缺数而中断。
        - 不重试：重试策略交给更上层（如 CLI 层按 error 字段选择性重跑），
          避免在本层重复引入网络重试逻辑。
        - 结果列表与输入 codes 一一对应，方便调用方按索引回填。
    """
    results = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            fig, report, df = run_one(code, start, end, style, provider, refresh)
            results.append({"code": code, "ok": True, "fig": fig, "report": report, "df": df, "error": None})
        except Exception as exc:
            # 失败留档：保留 code 与 error，其他字段置 None，便于后续统一处理
            results.append({"code": code, "ok": False, "fig": None, "report": None, "df": None, "error": str(exc)})
        # 每处理完一只就回调一次，让上层能实时刷新进度（而不是等整批结束）
        if progress_cb:
            progress_cb(i, total, code, results[-1]["ok"])
    return results
