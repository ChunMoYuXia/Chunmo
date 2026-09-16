"""提示词组装：指标表快照 → 结构化提示词（6 段专业研报）。

本模块职责单一：把 DataFrame 中的技术指标抽成「事实字典」，再按
6 段式模板格式化为自然语言片段，最后与外部提示词模板拼接，
交给 client.ask_llm 调用。

为什么把事实抽取（snapshot_facts）与提示词拼装（build_prompt）拆开？
- snapshot_facts 产出的字典是「AI 提示词」与「模板兜底报告」共用的
  单一事实源（Single Source of Truth），避免两处各自从 DataFrame 取值
  导致口径不一致。
- build_prompt 只负责把事实字典渲染成文本，便于单元测试时直接构造
  假字典验证渲染逻辑。
"""
from ..config import load_prompt_template


def snapshot_facts(df) -> dict:
    """最新一行快照 → 事实字典（AI 提示词与模板兜底共用）。

    参数:
        df: 按日期升序排列的指标 DataFrame，至少包含 date/close/volume/
            rsi14/dif/dea/macd 等列；ma5/ma10/ma20、k/d、boll_* 为可选列。

    返回:
        包含 20+ 键的事实字典，键名即含义（close/pct1/ma_arrange/...）。

    容错策略:
        - 若 df 不足 2 行，prev 退化为 last，pct1 记为 0。
        - 若不足 6 行，pct5 记为 0；vol_avg5 退化为全量均值。
        - 量比分母为 0 时记为 1.0（中性）。
        - KDJ 列不存在时，kdj_cross 记为「—」、kdj 为 None。
        - 布林列不存在时，用 .get + 默认值（inf/0）保证 boll_pos 判定不崩。
    """
    last = df.iloc[-1]
    # prev 取倒数第二行；不足 2 行时退化为 last，保证 pct1 计算不除零
    prev = df.iloc[-2] if len(df) >= 2 else last
    # 量比：当日成交量 / 近 5 日平均成交量（不含当日），衡量放量/缩量程度
    vol_avg5 = df["volume"].iloc[-6:-1].mean() if len(df) >= 6 else df["volume"].mean()
    vol_ratio = last["volume"] / vol_avg5 if vol_avg5 > 0 else 1.0
    # 当日涨跌幅（%），prev close 为 0 时记为 0 避免除零
    pct1 = (last["close"] / prev["close"] - 1) * 100 if prev["close"] else 0
    # 近 5 日涨跌幅（%），不足 6 行记为 0
    pct5 = (last["close"] / df.iloc[-6]["close"] - 1) * 100 if len(df) >= 6 else 0
    # MACD 金叉死叉判断：需对比今日与昨日的 DIF/DEA 交叉关系
    # 顺序：金叉（上穿）→ 死叉（下穿）→ 多头（DIF>DEA）→ 空头
    macd_cross = "金叉" if last["dif"] > last["dea"] and prev["dif"] <= prev["dea"] else (
        "死叉" if last["dif"] < last["dea"] and prev["dif"] >= prev["dea"] else "多头" if last["dif"] > last["dea"] else "空头")
    # KDJ 金叉死叉：仅当 k/d 列存在时计算，否则返回占位符
    if "k" in df.columns and "d" in df.columns:
        kdj_cross = "金叉" if last["k"] > last["d"] and prev["k"] <= prev["d"] else (
            "死叉" if last["k"] < last["d"] and prev["k"] >= prev["d"] else "多头" if last["k"] > last["d"] else "空头")
        kdj_val = last["k"]
    else:
        kdj_cross = "—"
        kdj_val = None
    # 均线排列：MA5>MA10>MA20 为多头，反之为空头，其余为纠缠
    ma_vals = {}
    for w in (5, 10, 20):
        col = f"ma{w}"
        if col in df.columns:
            ma_vals[w] = last[col]
    # 用 .get(..., 0) 兜底缺失列，保证链式比较不出错
    ma_bull = ma_vals.get(5, 0) > ma_vals.get(10, 0) > ma_vals.get(20, 0)
    ma_bear = ma_vals.get(5, 0) < ma_vals.get(10, 0) < ma_vals.get(20, 0)
    ma_arrange = "多头排列" if ma_bull else ("空头排列" if ma_bear else "纠缠")

    return {
        "date": last["date"].date(),
        "close": last["close"],
        "pct1": pct1,
        "volume": last["volume"],
        "vol_ratio": vol_ratio,
        "ma5": ma_vals.get(5),
        "ma10": ma_vals.get(10),
        "ma20": ma_vals.get(20),
        "ma_arrange": ma_arrange,
        "up5": "之上" if last["close"] >= ma_vals.get(5, last["close"]) else "之下",
        "up10": "之上" if last["close"] >= ma_vals.get(10, last["close"]) else "之下",
        "up20": "之上" if last["close"] >= ma_vals.get(20, last["close"]) else "之下",
        "pct5": pct5,
        "hi": df["close"].iloc[-20:].max(),
        "lo": df["close"].iloc[-20:].min(),
        "rsi": last["rsi14"],
        "rsi_zone": "超买" if last["rsi14"] > 70 else ("超卖" if last["rsi14"] < 30 else "中性"),
        "dif": last["dif"],
        "dea": last["dea"],
        "macd": last["macd"],
        "macd_cross": macd_cross,
        "kdj": kdj_val,
        "kdj_cross": kdj_cross,
        "boll_upper": last.get("boll_upper"),
        "boll_lower": last.get("boll_lower"),
        "boll_mid": last.get("boll_mid"),
        # 布林带位置：上轨之上（强势突破）/ 下轨之下（弱势破位）/ 轨道内
        "boll_pos": "上轨之上" if last["close"] > last.get("boll_upper", float("inf")) else (
            "下轨之下" if last["close"] < last.get("boll_lower", 0) else "轨道内"),
    }


def build_prompt(df, code: str, style: str = "要点式") -> str:
    """提示词 = 模板文件 + 6 段事实填充。

    参数:
        df: 指标数据表。
        code: 股票代码，会替换模板中的 {code} 占位符。
        style: 输出风格，支持「要点式」（短要点）和「成段」（分析师文章），
               决定模板中 {style_req} 的具体措辞。

    返回:
        完整的提示词文本，可直接交给 ask_llm。

    占位符说明（在外部模板文件中定义）:
        {code}      —— 股票代码
        {date}      —— 最新交易日
        {facts}     —— 6 段结构化指标事实（本函数拼装）
        {style_req} —— 风格要求文案（要点式 / 成段）

    为什么用 str.format 而非 f-string？
        模板存放在外部可编辑文件中，运行时才加载，
        必须用 format 在运行时替换占位符，f-string 无法做到。
    """
    f = snapshot_facts(df)
    # 6 段事实片段：行情概览 / 趋势研判 / 量价分析 / 技术指标 / 关键价位
    # （投资评级由 AI 自行生成，模板兜底时由 _rating_from_facts 补全）
    facts = (
        f"【行情概览】\n"
        f"- 收盘价 {f['close']:.2f}，当日涨跌 {f['pct1']:+.2f}%\n"
        f"- 成交量 {f['volume']:.0f}，量比 {f['vol_ratio']:.2f}（>1.5 放量，<0.7 缩量）\n"
        f"【趋势研判】\n"
        f"- MA5={f['ma5']:.2f}，MA10={f['ma10']:.2f}，MA20={f['ma20']:.2f}，均线{ f['ma_arrange']}\n"
        f"- 收盘价位于 MA5 {f['up5']}、MA10 {f['up10']}、MA20 {f['up20']}\n"
        f"- 布林带位置：{f['boll_pos']}（上轨 {f['boll_upper']:.2f} / 中轨 {f['boll_mid']:.2f} / 下轨 {f['boll_lower']:.2f}）\n"
        f"【量价分析】\n"
        f"- 近 5 个交易日涨跌 {f['pct5']:+.2f}%，量比 {f['vol_ratio']:.2f}\n"
        f"- 价量关系：{'价涨量增，多头积极' if f['pct1'] > 0 and f['vol_ratio'] > 1 else '价涨量缩，上攻乏力' if f['pct1'] > 0 else '价跌量增，空头积极' if f['vol_ratio'] > 1 else '价跌量缩，抛压减轻'}\n"
        f"【技术指标】\n"
        f"- RSI14 = {f['rsi']:.1f}（{f['rsi_zone']}）\n"
        f"- MACD：DIF {f['dif']:.2f} / DEA {f['dea']:.2f}，{f['macd_cross']}，柱值 {f['macd']:.2f}\n"
        f"- KDJ：K={f['kdj']:.1f}，{f['kdj_cross']}\n"
        f"【关键价位】\n"
        f"- 压力位：近 20 日最高 {f['hi']:.2f} / 布林上轨 {f['boll_upper']:.2f}\n"
        f"- 支撑位：近 20 日最低 {f['lo']:.2f} / 布林下轨 {f['boll_lower']:.2f}\n"
    )
    # 风格要求：把 style 枚举映射为自然语言指令，让 AI 按指定格式输出
    style_req = {
        "要点式": "用简短要点输出，每条一行，带小标题",
        "成段": "写成连贯的分析师文章段落",
    }[style]
    template = load_prompt_template()
    # 运行时替换四个占位符，产出最终提示词
    return template.format(code=code, date=f["date"], facts=facts, style_req=style_req)
