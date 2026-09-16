"""Streamlit 网页入口：研报 / 对比 / 预测 / 设置 / 日志 五个模块。

启动：python run.py

本模块采用「顶层脚本 + if/elif 路由」结构：每次 rerun 都会从上到下重新执行，
页面状态完全由 session_state 持久化。选择这种结构而非多页 app（pages/）
是为了让侧边栏的全局控件（默认模型、强制刷新缓存）能在所有模块间共享。
"""
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# 把项目根目录加入 sys.path，使 `quantai` 包可被直接 import。
# 用 parents[3] 是因为本文件位于 src/quantai/web/app.py，上溯三级到项目根。
# 放 import 之前是为了让后续 from quantai... 能正常解析。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quantai import audit
from quantai.ai.client import ask_llm
from quantai.config import (
    STOCK_POOL,
    all_groups,
    load_compare_template,
    load_custom_groups,
    load_prompt_template,
    load_providers,
    reset_prompt_template,
    save_custom_groups,
    save_keys,
    save_prompt_template,
    save_providers,
)
from quantai.data.cache import cache_stats, clear_cache, list_cached_stocks
from quantai.ml.models import ALL_MODEL_NAMES, get_models
from quantai.ml.predictor import analyze_stock
from quantai.report.batch import run_batch
from quantai.report.chart import make_compare_chart
from quantai.report.compare import compare_data, gen_compare_report
from quantai.report.export import export_compare_html, export_html, export_pdf
from quantai.settings import (
    DEFAULT_SETTINGS,
    load_settings,
    reset_settings,
    save_settings,
)

# 页面标题与布局放在所有业务逻辑之前，保证每个页面都使用统一配置
st.set_page_config(page_title="QuantAI 量化研报", layout="wide", page_icon="📈")

# ── 侧边栏导航与全局状态 ───────────────────────────────────
# 侧边栏承载「全局共享控件」：模块选择、默认模型、是否强制刷新缓存。
# 之所以不放进各页面内部，是为了切换模块时这些参数不丢失，
# 且 run_batch/compare_data/analyze_stock 都需要 provider / refresh。
with st.sidebar:
    st.title("📈 QuantAI")
    st.caption("量化研报 + 多模型预测一体化")
    # radio 用 key 默认按 label 生成；index=0 让首次进入落在「研报生成」
    page = st.radio("功能模块", ["研报生成", "分组对比", "多模型预测", "设置中心", "操作日志"], index=0)
    st.divider()
    providers = load_providers()
    # 默认大模型：所有调用 ask_llm 的页面都复用此变量，避免每页重复选择
    provider = st.selectbox("默认大模型", list(providers), key="global_provider")
    # 强制刷新：默认 False 走缓存提速，用户主动勾选时绕过缓存取最新行情
    refresh = st.checkbox("强制刷新数据缓存", value=False)
    st.caption(f"共 {len(providers)} 个模型供应商")

# 构造「代码 → 名称」的全局映射：先内置股票池，再叠加用户自定义组，
# 这样在任何页面都能通过 NAMES[code] 查到中文名（自定义组优先覆盖）。
NAMES = {c: n for group in STOCK_POOL.values() for c, n in group.items()}
# 合并自定义组的名称
for g in load_custom_groups().values():
    NAMES.update(g)


# ──────────────────────────────────────────────────────────
# 研报生成
# ──────────────────────────────────────────────────────────
if page == "研报生成":
    st.header("📊 AI 研报生成器")
    st.caption("输入股票代码 → 自动下载行情 → 技术指标 → 交互式 K 线图 → AI 六段式股评")

    # ── pending_codes 跨 rerun 状态保持技巧 ──
    # Streamlit 按钮点击后会 rerun，但 text_input 的值是由 key 绑定的，
    # 直接修改 codes_text 不会反映到 UI。这里用「中转变量」方案：
    # 1. 按钮把要追加的代码写入 pending_codes；
    # 2. 下次 rerun 时把 pending_codes 回填到 codes_text，再清空 pending_codes。
    # 这样能在不破坏 text_input 受控状态的前提下实现「追加代码」。
    if "pending_codes" not in st.session_state:
        st.session_state["pending_codes"] = None
    if st.session_state["pending_codes"] is not None:
        # 回填到 text_input 的 key，使其立即显示新值
        st.session_state["codes_text"] = st.session_state["pending_codes"]
        st.session_state["pending_codes"] = None
    if "codes_text" not in st.session_state:
        st.session_state["codes_text"] = "600519"

    col_input, col_pick = st.columns([2, 1])
    with col_input:
        codes_text = st.text_input("股票代码（逗号分隔，可多只）", key="codes_text")
    with col_pick:
        # 自选池来自内置分组 + 自定义组的展平列表，格式「代码 名称」
        all_stocks = [(c, n) for g in all_groups().values() for c, n in g.items()]
        pick = st.selectbox("自选池", [f"{c} {n}" for c, n in all_stocks])
        if st.button("➕ 加入"):
            codes = [c.strip() for c in codes_text.split(",") if c.strip()]
            chosen = pick.split()[0]
            # 去重：已在列表中的代码不再追加，避免重复生成
            if chosen not in codes:
                codes.append(chosen)
            st.session_state["pending_codes"] = ",".join(codes)
            st.rerun()

    # 按分组批量加入：最多 5 列按钮，点击把整组代码去重追加
    with st.expander("📁 按分组一键加入", expanded=False):
        groups = all_groups()
        cols = st.columns(min(len(groups), 5))
        for col, gname in zip(cols, list(groups.keys())[:5]):
            with col:
                if st.button(gname, key=f"btn_{gname}"):
                    codes = [c.strip() for c in codes_text.split(",") if c.strip()]
                    for c in groups[gname]:
                        if c not in codes:
                            codes.append(c)
                    st.session_state["pending_codes"] = ",".join(codes)
                    st.rerun()

    # 最终解析：统一按逗号切分并去空白，作为批量处理的输入
    codes = [c.strip() for c in codes_text.split(",") if c.strip()]
    st.caption(f"解析出 {len(codes)} 只：{'、'.join(codes)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        # 默认回看一年，平衡数据量与计算速度
        start = st.date_input("开始日期", value=dt.date.today() - dt.timedelta(days=365))
    with col2:
        end = st.date_input("结束日期", value=dt.date.today())
    with col3:
        style = st.radio("股评样式", ["要点式", "成段"], horizontal=True)

    if st.button("🚀 生成批量研报", type="primary"):
        if not codes:
            st.error("请至少输入一只股票代码")
        else:
            t0 = time.time()
            # 进度条：用回调函数实时更新当前处理到第几只、是否成功
            progress = st.progress(0, text=f"准备处理 {len(codes)} 只……")

            def on_progress(i, total, code, ok):
                """批量处理进度回调。

                Args:
                    i: 已完成数量（含当前）
                    total: 总数
                    code: 当前处理的股票代码
                    ok: 当前股票是否生成成功
                直接操作 progress 组件是因为 Streamlit 回调内不允许 st.rerun，
                但可以更新已创建的 progress 元素。
                """
                progress.progress(i / total, text=f"{i}/{total} {code} {'✅' if ok else '❌'}")

            results = run_batch(
                codes, start.isoformat(), end.isoformat(),
                style, provider, refresh=refresh, progress_cb=on_progress,
            )
            # 处理完成后清空进度条，避免残留
            progress.empty()
            ok_n = sum(r["ok"] for r in results)
            duration = int((time.time() - t0) * 1000)
            st.success(f"完成：成功 {ok_n} / 共 {len(results)}（耗时 {duration}ms）")
            # 审计日志：记录批量操作的整体结果，用于操作日志页追溯
            audit.record("生成研报", user_input={"codes": codes, "start": start.isoformat(), "end": end.isoformat()},
                         result="success", duration_ms=duration, success_count=ok_n, total=len(results))

            # 逐只展示结果：成功的展开显示图表 + 研报 + 下载按钮，失败的展示错误
            for r in results:
                mark = "✅" if r["ok"] else "❌"
                title = f"{r['code']} {NAMES.get(r['code'], '')} {mark}"
                # 成功的默认展开，失败的默认折叠，提升浏览效率
                with st.expander(title, expanded=r["ok"]):
                    if r["ok"]:
                        st.plotly_chart(r["fig"], use_container_width=True)
                        st.markdown(r["report"])
                        col_h, col_p = st.columns(2)
                        with col_h:
                            # HTML 导出：依赖 plotly 的 to_html，无需额外依赖
                            html_path = export_html(r["fig"], r["report"], r["code"], NAMES.get(r["code"], ""))
                            with open(html_path, "rb") as fh:
                                st.download_button("📥 下载 HTML", fh.read(), file_name=html_path.name, mime="text/html")
                        with col_p:
                            # PDF 导出：需要 weasyprint 或 reportlab，缺失时优雅降级为提示
                            pdf_path = export_pdf(r["fig"], r["report"], r["code"], NAMES.get(r["code"], ""))
                            if pdf_path:
                                with open(pdf_path, "rb") as fh:
                                    st.download_button("📄 下载 PDF", fh.read(), file_name=pdf_path.name, mime="application/pdf")
                            else:
                                st.caption("PDF 导出需安装 weasyprint 或 reportlab")
                    else:
                        st.error(r["error"])
                        # 失败也单独记一条审计日志，便于定位单只股票失败原因
                        audit.record("生成研报", user_input={"code": r["code"]}, result="failed", error=r["error"])


# ──────────────────────────────────────────────────────────
# 分组对比
# ──────────────────────────────────────────────────────────
elif page == "分组对比":
    st.header("🔀 分组对比")
    st.caption("两组股票归一化收益率同屏对比 + AI 对比研报。支持自定义组和临时输入。")

    groups = all_groups()
    gnames = list(groups.keys())

    # 模式选择：两种模式互斥。
    # 「选择已有组」直接用 all_groups() 里的定义，名称可追溯；
    # 「自定义输入」让用户临时输入两组代码做一次性对比，不持久化。
    mode = st.radio("对比模式", ["选择已有组", "自定义输入"], horizontal=True)

    if mode == "选择已有组":
        col_a, col_b = st.columns(2)
        with col_a:
            # 组 1 默认取第一个分组
            group1_name = st.selectbox("组 1", gnames, index=0, key="group1")
        with col_b:
            # 组 2 默认取第二个分组；若只有一个分组则兜底取第一个，避免索引越界
            group2_name = st.selectbox("组 2", gnames, index=min(1, len(gnames) - 1), key="group2")
        group1 = groups[group1_name]
        group2 = groups[group2_name]
    else:
        # 自定义模式：按逗号切分代码，名称从全局 NAMES 查，查不到则用代码本身当名称
        st.caption("输入两组股票代码（逗号分隔），名称可留空。")
        col_a, col_b = st.columns(2)
        with col_a:
            g1_codes = st.text_input("组 1 代码", value="600519,000858", key="g1_codes")
        with col_b:
            g2_codes = st.text_input("组 2 代码", value="601318,600036", key="g2_codes")
        # 字典推导：{code: name}，与 all_groups() 返回结构一致，方便后续统一处理
        group1 = {c.strip(): NAMES.get(c.strip(), c.strip()) for c in g1_codes.split(",") if c.strip()}
        group2 = {c.strip(): NAMES.get(c.strip(), c.strip()) for c in g2_codes.split(",") if c.strip()}

    # 保存自定义组：把当前组 1 / 组 2 持久化到自定义组文件，
    # 下次进入时 all_groups() 会自动包含，方便重复对比。
    with st.expander("💾 保存当前组合为自定义组", expanded=False):
        save_name = st.text_input("组名称", key="save_group_name")
        if st.button("保存组 1") and group1:
            custom = load_custom_groups()
            # 留空时用默认名「自定义组1」，避免空字符串键覆盖已有项
            custom[save_name or "自定义组1"] = group1
            save_custom_groups(custom)
            st.success("已保存，刷新后可选")
        if st.button("保存组 2") and group2:
            custom = load_custom_groups()
            custom[save_name or "自定义组2"] = group2
            save_custom_groups(custom)
            st.success("已保存，刷新后可选")

    col1, col2 = st.columns(2)
    with col1:
        # 对比默认回看半年：太短趋势不明显，太长则受牛熊切换干扰
        c_start = st.date_input("开始日期", value=dt.date.today() - dt.timedelta(days=180), key="c_start")
    with col2:
        c_end = st.date_input("结束日期", value=dt.date.today(), key="c_end")

    if st.button("🔍 生成对比", type="primary"):
        if not group1 or not group2:
            st.error("两组都不能为空")
        else:
            # 标签：已有组用组名，自定义组用占位名，便于图表与研报区分两组
            g1_label = group1_name if mode == "选择已有组" else "自定义组1"
            g2_label = group2_name if mode == "选择已有组" else "自定义组2"
            groups_dict = {g1_label: group1, g2_label: group2}
            t0 = time.time()
            with st.spinner("下载两组数据、归一化、写对比研报……"):
                curves, means, table = compare_data(
                    groups_dict, c_start.isoformat(), c_end.isoformat(), refresh=refresh,
                )
                fig = make_compare_chart(curves, means)
                # AI 对比研报可能因模型/网络失败，用 try/except 降级为文本提示，
                # 保证图表和数据表仍能正常展示，不阻塞核心功能
                try:
                    ai_compare = gen_compare_report(
                        groups_dict, curves, c_start.isoformat(), c_end.isoformat(), provider,
                    )
                except Exception as exc:
                    ai_compare = f"（AI 对比研报生成失败，已降级：{exc}）"
            duration = int((time.time() - t0) * 1000)
            audit.record("分组对比", user_input={"group1": g1_label, "group2": g2_label},
                         result="success", duration_ms=duration)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table, use_container_width=True)
            st.markdown("#### AI 对比研报")
            st.write(ai_compare)
            # 导出对比 HTML：包含图表、数据表、AI 研报，方便离线分享
            html_path = export_compare_html(fig, table, ai_compare, g1_label, g2_label)
            with open(html_path, "rb") as fh:
                st.download_button("📥 下载对比 HTML", fh.read(), file_name=html_path.name, mime="text/html")


# ──────────────────────────────────────────────────────────
# 多模型预测
# ──────────────────────────────────────────────────────────
elif page == "多模型预测":
    st.header("🤖 多模型预测与智能决策")
    st.caption("支持选择多个模型（逻辑回归/XGBoost/LightGBM/MLP/LSTM）时序赛跑，冠军预测上涨概率")

    # 模型多选：从注册表取出所有可用模型名，
    # 默认值取自 settings.ml.default_models，但用列表推导过滤掉已被删除的模型，
    # 避免 default 里出现不在 options 中的项导致 Streamlit 报错。
    available_models = list(get_models().keys())
    selected_models = st.multiselect(
        "选择参赛模型（可多选）", available_models,
        default=[m for m in load_settings()["ml"]["default_models"] if m in available_models],
    )
    # 至少要选一个模型，否则 analyze_stock 没有候选可赛跑
    if not selected_models:
        st.warning("请至少选择一个模型")

    pred_text = st.text_input("股票代码（逗号分隔）", value="600519", key="pred_text")
    col1, col2 = st.columns(2)
    with col1:
        # 默认回看两年：机器学习模型需要足够的训练样本，两年比较稳妥
        p_start = st.date_input("开始日期", value=dt.date.today() - dt.timedelta(days=365 * 2), key="p_start")
    with col2:
        p_end = st.date_input("结束日期", value=dt.date.today(), key="p_end")

    # selected_models 为空时按钮虽然显示但不执行，避免空列表调用
    if st.button("🎯 生成预测", type="primary") and selected_models:
        pred_codes = [c.strip() for c in pred_text.split(",") if c.strip()]
        if not pred_codes:
            st.error("请至少输入一只股票代码")
        else:
            t0 = time.time()
            progress = st.progress(0, text=f"训练 {len(selected_models)} 个模型 × {len(pred_codes)} 只股票……")
            preds = []
            # 逐只股票训练预测：每只内部会训练所选全部模型并排序选冠军
            for i, c in enumerate(pred_codes, 1):
                try:
                    preds.append(analyze_stock(
                        c, p_start.isoformat(), p_end.isoformat(), provider,
                        model_names=selected_models,
                    ))
                except Exception as exc:
                    # 单只失败不影响其他股票，把错误信息塞进结果列表统一展示
                    preds.append({"code": c, "error": str(exc)})
                progress.progress(i / len(pred_codes), text=f"{i}/{len(pred_codes)} {c}")
            progress.empty()
            duration = int((time.time() - t0) * 1000)
            audit.record("多模型预测", user_input={"codes": pred_codes, "models": selected_models},
                         result="success", duration_ms=duration)

            # 逐只展示：失败的单独折叠显示错误，成功的展示模型排名 + 上涨概率 + AI 建议
            for r in preds:
                if "error" in r:
                    with st.expander(f"{r['code']} ❌"):
                        st.error(r["error"])
                    audit.record("多模型预测", user_input={"code": r["code"]}, result="failed", error=r["error"])
                    continue
                # 标题带上冠军模型名与上涨概率，让用户一眼看到结论
                with st.expander(
                    f"{r['code']} {NAMES.get(r['code'], '')} · 冠军 {r['best']} · "
                    f"上涨概率 {r['prob_up']:.1%}", expanded=True,
                ):
                    # 排名表：展示每个模型的 LogLoss 和 AUC，方便对比各模型表现
                    rank_df = pd.DataFrame([
                        {"模型": name, "LogLoss": f"{s['log_loss']:.4f}", "AUC": f"{s['auc']:.3f}"}
                        for name, s in r["ranking"]
                    ])
                    st.table(rank_df)
                    prob = r["prob_up"]
                    # 用进度条直观展示上涨概率
                    st.progress(prob, text=f"上涨概率 {prob:.1%}")
                    st.markdown("#### AI 风控建议")
                    st.write(r["advice"])


# ──────────────────────────────────────────────────────────
# 设置中心
# ──────────────────────────────────────────────────────────
elif page == "设置中心":
    st.header("⚙️ 设置中心")
    # 每次进入设置页都重新加载，保证展示的是磁盘上的最新值
    settings = load_settings()

    # 9 个标签页：技术指标 / 图表显示 / 研报样式 / 机器学习 / 数据源 /
    # 提示词模板 / 模型注册表 / API 钥匙 / 数据缓存
    tab_ind, tab_chart, tab_report, tab_ml, tab_data, tab_prompt, tab_provider, tab_keys, tab_cache = st.tabs([
        "指标参数", "图表显示", "研报样式", "机器学习", "数据源", "提示词模板", "模型注册表", "API 钥匙", "数据缓存",
    ])

    # ── 标签页 1：技术指标参数 ──
    with tab_ind:
        st.subheader("技术指标参数")
        ind = settings["indicators"]
        col1, col2 = st.columns(2)
        with col1:
            # 均线窗口用逗号分隔的字符串输入，方便一次改多个窗口
            ma_str = st.text_input("均线窗口（逗号分隔）", ",".join(map(str, ind["ma_windows"])))
            rsi_w = st.number_input("RSI 周期", value=ind["rsi_window"], min_value=2, max_value=60)
            macd_f = st.number_input("MACD 快线", value=ind["macd_fast"], min_value=2)
            macd_s = st.number_input("MACD 慢线", value=ind["macd_slow"], min_value=2)
        with col2:
            macd_sig = st.number_input("MACD 信号线", value=ind["macd_signal"], min_value=1)
            boll_w = st.number_input("布林带窗口", value=ind["boll_window"], min_value=2)
            boll_std = st.number_input("布林带标准差倍数", value=ind["boll_std"], min_value=0.5, max_value=5.0, step=0.1)
            kdj_n = st.number_input("KDJ N", value=ind["kdj_n"], min_value=2)
        if st.button("💾 保存指标参数", key="save_ind"):
            try:
                # ma_windows 是字符串需手动转 int 列表；其余由 number_input 保证类型
                settings["indicators"]["ma_windows"] = [int(x.strip()) for x in ma_str.split(",") if x.strip()]
                settings["indicators"]["rsi_window"] = int(rsi_w)
                settings["indicators"]["macd_fast"] = int(macd_f)
                settings["indicators"]["macd_slow"] = int(macd_s)
                settings["indicators"]["macd_signal"] = int(macd_sig)
                settings["indicators"]["boll_window"] = int(boll_w)
                settings["indicators"]["boll_std"] = float(boll_std)
                settings["indicators"]["kdj_n"] = int(kdj_n)
                save_settings(settings)
                st.success("已保存")
            except ValueError:
                # 均线窗口若填了非数字会进这里，提示用户修正
                st.error("参数格式错误")

    # ── 标签页 2：图表显示 ──
    with tab_chart:
        st.subheader("图表显示")
        ch = settings["chart"]
        col1, col2 = st.columns(2)
        with col1:
            # 各副图开关：不想要的可以关掉，减少图表拥挤
            ch["show_ma"] = st.checkbox("显示均线", value=ch["show_ma"])
            ch["show_boll"] = st.checkbox("显示布林带", value=ch["show_boll"])
            ch["show_volume"] = st.checkbox("成交量副图", value=ch["show_volume"])
            ch["show_rsi"] = st.checkbox("RSI 副图", value=ch["show_rsi"])
        with col2:
            ch["show_macd"] = st.checkbox("MACD 副图", value=ch["show_macd"])
            ch["show_kdj"] = st.checkbox("KDJ 副图", value=ch["show_kdj"])
            # 图表高度按 50 步进，避免过细
            ch["height"] = st.slider("图表高度", min_value=500, max_value=1200, value=ch["height"], step=50)
        if st.button("💾 保存图表设置", key="save_chart"):
            settings["chart"] = ch
            save_settings(settings)
            st.success("已保存")

    # ── 标签页 3：研报样式 ──
    with tab_report:
        st.subheader("研报样式")
        rp = settings["report"]
        # 字数上限：避免 LLM 输出过长影响速度和阅读
        rp["max_words"] = st.number_input("股评字数上限", value=rp["max_words"], min_value=50, max_value=1000, step=50)
        # 默认样式与研报生成页的 radio 保持一致
        rp["style"] = st.radio("默认样式", ["要点式", "成段"], index=0 if rp["style"] == "要点式" else 1)
        rp["include_rating"] = st.checkbox("输出投资评级", value=rp["include_rating"])
        if st.button("💾 保存研报样式", key="save_report"):
            settings["report"] = rp
            save_settings(settings)
            st.success("已保存")

    # ── 标签页 4：机器学习 ──
    with tab_ml:
        st.subheader("机器学习")
        ml = settings["ml"]
        available = list(get_models().keys())
        # 默认参赛模型：同样过滤掉已删除的模型，避免 default 越界
        ml["default_models"] = st.multiselect(
            "默认参赛模型", available, default=[m for m in ml["default_models"] if m in available],
        )
        # 时序交叉验证折数：时序数据不能随机打乱，用 TimeSeriesSplit
        ml["n_splits"] = st.number_input("时序交叉验证折数", value=ml["n_splits"], min_value=2, max_value=10)
        col1, col2, col3 = st.columns(3)
        with col1:
            ml["lstm_hidden"] = st.number_input("LSTM 隐藏层维度", value=ml["lstm_hidden"], min_value=8, max_value=256)
        with col2:
            ml["lstm_layers"] = st.number_input("LSTM 层数", value=ml["lstm_layers"], min_value=1, max_value=5)
        with col3:
            # 训练轮数上限 500，防止过拟合 + 训练过久
            ml["lstm_epochs"] = st.number_input("LSTM 训练轮数", value=ml["lstm_epochs"], min_value=10, max_value=500)
        if st.button("💾 保存 ML 设置", key="save_ml"):
            settings["ml"] = ml
            save_settings(settings)
            st.success("已保存")

    # ── 标签页 5：数据源 ──
    with tab_data:
        st.subheader("数据源")
        # 主数据源：baostock 稳定但延迟，akshare 实时但偶有风控，二选一
        settings["data"]["primary"] = st.selectbox(
            "主数据源", ["baostock", "akshare"],
            index=0 if settings["data"]["primary"] == "baostock" else 1,
        )
        # 缓存开关 + TTL：开启后同一只股票在有效期内不再重复下载
        settings["cache"]["enabled"] = st.checkbox("启用数据缓存", value=settings["cache"]["enabled"])
        settings["cache"]["ttl_days"] = st.number_input(
            "缓存有效期（天）", value=settings["cache"]["ttl_days"], min_value=0, max_value=365,
        )
        if st.button("💾 保存数据源设置", key="save_data"):
            save_settings(settings)
            st.success("已保存")

    # ── 标签页 6：提示词模板 ──
    with tab_prompt:
        st.subheader("股评提示词模板")
        # 提示词占位符 {code} {date} {facts} {style_req} 是模板与业务代码的契约，
        # 删了会导致渲染失败，所以明确提醒用户
        template = st.text_area(
            "全文可改；{code} {date} {facts} {style_req} 四个占位符勿删",
            value=load_prompt_template(), height=300,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 保存提示词"):
                save_prompt_template(template)
                st.success("已保存")
        with col_b:
            # 恢复出厂：直接重置为内置模板，然后 rerun 让 text_area 刷新显示
            if st.button("🔄 恢复出厂提示词"):
                reset_prompt_template()
                st.success("已恢复")
                st.rerun()

    # ── 标签页 7：模型注册表 ──
    with tab_provider:
        st.subheader("模型注册表")
        providers = load_providers()
        # 只读展示已有模型的配置
        for name, (url, model, key_env) in providers.items():
            st.write(f"- **{name}**：`{url}` ／ 模型 `{model}` ／ 钥匙 `{key_env}`")
        with st.expander("➕ 新增模型"):
            new_name = st.text_input("名字", key="new_name")
            new_url = st.text_input("接口网址", key="new_url")
            new_model = st.text_input("模型名", key="new_model")
            new_key_env = st.text_input("钥匙环境变量名", key="new_key_env")
            if st.button("添加模型"):
                # 四个字段必填，否则无法调用
                if all([new_name, new_url, new_model, new_key_env]):
                    if new_name in providers:
                        st.error("名字已存在")
                    else:
                        # 用 list 而非 tuple，因为 save/load 走 JSON 会变成 list
                        providers[new_name] = [new_url, new_model, new_key_env]
                        save_providers(providers)
                        st.success("已添加")
                else:
                    st.error("四个字段都要填")
        with st.expander("🗑️ 删除模型"):
            del_name = st.selectbox("选择要删除的", list(providers), key="del_name")
            if st.button("删除模型"):
                # 至少保留一家供应商，否则研报/对比/预测都无法调用 LLM
                if len(providers) > 1:
                    del providers[del_name]
                    save_providers(providers)
                    st.success("已删除")
                else:
                    st.error("至少保留一家")

    # ── 标签页 8：API 钥匙 ──
    with tab_keys:
        st.subheader("API 钥匙")
        # 安全设计：留空表示不修改；密码框不回显，且 save_keys 内部不会读出原值
        st.caption("留空 = 不改；钥匙只进不出，不会回显。")
        with st.form("keys_form"):
            # 从所有 provider 的 key_env 去重排序，避免同一环境变量重复出现
            key_inputs = {
                env: st.text_input(env, type="password")
                for env in sorted({key_env for _, _, key_env in load_providers().values()})
            }
            if st.form_submit_button("💾 保存钥匙"):
                save_keys(key_inputs)
                st.success("已保存")

    # ── 标签页 9：数据缓存管理 ──
    with tab_cache:
        st.subheader("数据缓存管理")
        stats = cache_stats()
        if not stats:
            st.info("暂无缓存数据")
        else:
            st.write(f"共 {len(stats)} 只股票有缓存")
            for code, info in stats.items():
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(f"**{code}** {NAMES.get(code, '')}")
                with col_b:
                    st.caption(f"{info['files']} 个文件 · {info['size_kb']} KB")
                with col_c:
                    # 单只删除：key 用代码保证唯一
                    if st.button("🗑️ 删除", key=f"del_cache_{code}"):
                        n = clear_cache(code)
                        st.success(f"已删除 {n} 个文件")
                        st.rerun()
        # 清空全部：谨慎操作，所以放在单只列表之后
        if st.button("🗑️ 清空全部缓存"):
            n = clear_cache()
            st.success(f"已清空 {n} 个缓存文件")
            st.rerun()

    st.divider()
    # 一键恢复出厂设置：会覆盖所有自定义配置，放在底部避免误触
    if st.button("🔄 恢复全部出厂设置"):
        reset_settings()
        st.success("已恢复全部出厂设置")
        st.rerun()


# ──────────────────────────────────────────────────────────
# 操作日志
# ──────────────────────────────────────────────────────────
elif page == "操作日志":
    st.header("📋 操作审计日志")
    st.caption("记录所有操作，便于追溯纠错")

    # 顶部统计卡片：总操作数、失败数、操作类型数
    s = audit.stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("总操作数", s["total"])
    col2.metric("失败数", s["failed"])
    col3.metric("操作类型数", len(s["by_action"]))

    # 操作类型分布：按 action 分组计数，方便看哪些操作最频繁
    if s["by_action"]:
        st.markdown("**操作类型分布**")
        for act, cnt in s["by_action"].items():
            st.caption(f"- {act}: {cnt} 次")

    st.divider()
    # 三维度过滤：操作类型 + 结果 + 显示条数
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        # 「全部」对应 None，传给 audit.query 表示不过滤
        filter_action = st.selectbox("按类型过滤", ["全部"] + list(s["by_action"].keys()), key="filter_action")
    with col_f2:
        filter_result = st.selectbox("按结果过滤", ["全部", "success", "failed"], key="filter_result")
    with col_f3:
        limit = st.slider("显示条数", 10, 500, 100, key="log_limit")

    action = None if filter_action == "全部" else filter_action
    result = None if filter_result == "全部" else filter_result
    # query 内部按时间倒序返回，limit 控制最多展示条数
    entries = audit.query(limit=limit, action=action, result=result)

    if entries:
        # 把 entries 列表转成 DataFrame，方便用 st.dataframe 展示
        # 结果字段用 ✅/❌ 图标替代原始字符串，更直观
        df = pd.DataFrame([
            {
                "时间": e["timestamp"],
                "操作": e["action"],
                "结果": "✅" if e["result"] == "success" else "❌",
                "耗时(ms)": e.get("duration_ms", ""),
                "输入": str(e.get("input", "")),
                "错误": e.get("error", ""),
            }
            for e in entries
        ])
        # hide_index=True 隐藏行号，让表格更干净
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无日志记录")

    # 清空日志：会删除全部审计记录，谨慎操作
    if st.button("🗑️ 清空日志"):
        n = audit.clear()
        st.success(f"已清空 {n} 条日志")
        st.rerun()
