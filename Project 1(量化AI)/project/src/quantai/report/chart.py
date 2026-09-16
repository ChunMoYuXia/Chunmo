"""绘图器：K 线主图 + 可配置副图（成交量/RSI/MACD/KDJ）。

设计要点：
- 副图组合完全由 settings 动态决定，避免为不同需求维护多份绘图函数。
- 主图固定占 50% 高度，剩余空间由已开启的副图均分，保证主图始终可读。
- 涨跌配色遵循 A 股习惯（红涨绿跌），通过模块级常量统一管理。
- 分组对比图单独提供 make_compare_chart，与单股图解耦以便复用配色常量。
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..settings import load_settings

# A 股习惯配色：上涨红、下跌绿；集中定义便于全站统一与后期换肤
UP_COLOR = "#e53935"
DOWN_COLOR = "#26a69a"


def make_chart(df):
    """根据 settings 动态决定副图并绘制单股研报图。

    参数：
        df: 已附加技术指标的行情 DataFrame，必须包含 date/open/high/low/close/volume；
            其余指标列（ma*/boll_*/rsi14/macd/dif/dea/k/d/j）按需存在，
            函数内部通过 `col in df.columns` 做存在性判断以容错。

    返回：
        plotly.graph_objects.Figure: 含主图与已开启副图的组合图。

    动态配置思路：
        遍历 settings["chart"] 中的布尔开关，逐个累加行数并记录副图名称；
        最后统一用 make_subplots 一次性布局，避免事后调整导致的轴对齐问题。
        这种「先收集再绘制」的写法比「每开一个副图就调用 add_subplot」更高效，
        也更容易控制行高比例。
    """
    cfg = load_settings()["chart"]
    rows = 1  # 主图
    subplots = []
    # 以下四个开关彼此独立，按固定顺序（成交量→RSI→MACD→KDJ）堆叠，
    # 顺序固定是为了让用户视觉上形成稳定预期
    if cfg["show_volume"]:
        rows += 1
        subplots.append("volume")
    if cfg["show_rsi"]:
        rows += 1
        subplots.append("rsi")
    if cfg["show_macd"]:
        rows += 1
        subplots.append("macd")
    if cfg["show_kdj"]:
        rows += 1
        subplots.append("kdj")

    # 行高分配：主图占一半，副图均分剩余空间；无副图时主图占满
    if rows == 1:
        row_heights = [1.0]
    else:
        main_h = 0.5
        sub_h = (1.0 - main_h) / (rows - 1)
        row_heights = [main_h] + [sub_h] * (rows - 1)

    # shared_xaxes=True 让所有子图共用时间轴，缩放/拖拽时整体联动
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # 主图：K 线 + 均线 + 布林带
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR, name="K线",
    ), row=1, col=1)
    # 均线：MA5/MA10/MA20 用不同颜色区分；只画已计算出的列，防止指标缺失报错
    if cfg["show_ma"]:
        for w, color in [(5, "#ff9800"), (10, "#9c27b0"), (20, "#2196f3")]:
            col = f"ma{w}"
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[col], name=f"MA{w}",
                    line=dict(width=1, color=color),
                ), row=1, col=1)
    # 布林带：上下轨之间填充半透明色带，直观展示价格波动区间
    if cfg["show_boll"] and "boll_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["boll_upper"], name="BOLL上轨",
            line=dict(width=1, color="#bdbdbd", dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["boll_lower"], name="BOLL下轨",
            line=dict(width=1, color="#bdbdbd", dash="dash"),
            fill="tonexty", fillcolor="rgba(189,189,189,0.1)",
        ), row=1, col=1)

    # 副图：按收集到的顺序从第 2 行依次绘制
    for i, sp in enumerate(subplots, start=2):
        if sp == "volume":
            # 成交量柱颜色与当日涨跌一致，便于与 K 线对照
            vol_colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(df["close"], df["open"])]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["volume"], name="成交量", marker_color=vol_colors,
            ), row=i, col=1)
        elif sp == "rsi":
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["rsi14"], name="RSI14",
                line=dict(width=1, color="#f57c00"),
            ), row=i, col=1)
            # 70/30 超买超卖分界线，业内通用阈值
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=i, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=i, col=1)
        elif sp == "macd":
            # MACD 柱：正值红色、负值绿色，与涨跌语义一致
            macd_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df["macd"]]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["macd"], name="MACD柱", marker_color=macd_colors,
            ), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["dif"], name="DIF", line=dict(width=1, color="#1976d2"),
            ), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["dea"], name="DEA", line=dict(width=1, color="#f57c00"),
            ), row=i, col=1)
        elif sp == "kdj" and "k" in df.columns:
            # KDJ 三线同源同色（仅线宽区分），避免颜色过多干扰主图
            fig.add_trace(go.Scatter(x=df["date"], y=df["k"], name="K", line=dict(width=1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["d"], name="D", line=dict(width=1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["j"], name="J", line=dict(width=1)), row=i, col=1)

    # 统一布局：隐藏 rangeslider（K 线自带滑块与副图时间轴冲突），图例置于顶部
    fig.update_layout(
        title=f"K线研报图（{df['date'].iloc[0].date()} 至 {df['date'].iloc[-1].date()}）",
        xaxis_rangeslider_visible=False,
        height=cfg.get("height", 900),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# 分组对比用配色：每组固定一种颜色，组内所有个股线条复用同色（细线），
# 组均值用同色粗线突出，方便一眼区分组别并定位组内强弱
GROUP_COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00"]


def make_compare_chart(curves, means):
    """归一化收益率对比图。

    参数：
        curves: {组名: {代码: Series}}，每个 Series 为已归一化（起点=0）的收益率序列。
        means:  {组名: Series}，每组个股在同一时间轴上的均值收益率，用于绘制粗线突出组趋势。

    返回：
        plotly.graph_objects.Figure: 含所有个股细线 + 每组均值粗线的对比图。

    设计说明：
        - 个股线条 showlegend=False，避免图例被几十只股票撑爆；
          只把「组均值」加入图例，组内个股靠 hover 时查看。
        - 起点对齐为 0 是在 compare._normalize 中完成的，
          此处直接消费归一化后的 Series，Y 轴用百分比格式展示。
        - GROUP_COLORS 取模复用，确保超过 4 组时仍有颜色分配但不再保证视觉唯一。
    """
    fig = go.Figure()
    for i, (gname, cs) in enumerate(curves.items()):
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        # 组内每只股票：细线，不进图例，hover 显示具体代码
        for code, s in cs.items():
            fig.add_trace(go.Scatter(
                x=s.index, y=s, name=f"{gname}·{code}",
                line=dict(width=1, color=color),
                showlegend=False, hovertemplate=f"{gname}·{code}<br>%{{y:.2%}}<extra></extra>",
            ))
        # 组均值：粗线，进入图例代表该组整体走势
        fig.add_trace(go.Scatter(
            x=means[gname].index, y=means[gname], name=f"{gname} 组均值",
            line=dict(width=4, color=color),
        ))
    fig.update_layout(
        title="两组归一化收益率对比（起点对齐为 0）",
        height=550,
        yaxis_tickformat=".0%",
    )
    return fig
