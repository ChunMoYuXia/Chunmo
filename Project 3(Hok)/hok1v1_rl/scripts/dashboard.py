"""HoK 1v1 训练实时看板 —— 零第三方服务端依赖。

启动后浏览器打开 http://localhost:8600（WSL2 下 Windows 浏览器可直接访问）。

数据源是训练脚本写的 metrics.jsonl（见 hok1v1_rl.metrics_store），
页面每 5 秒拉取一次并重建图表，训练过程中看板随指标动态变化。

运行::

    python scripts/dashboard.py                    # 默认端口 8600，读 logs/metrics.jsonl
    python scripts/dashboard.py --port 9000 --metrics logs/metrics.jsonl
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from hok1v1_rl.metrics_store import MetricsStore     # noqa: E402

# ---------------------------------------------------------------------------
# 页面（单文件 HTML + Chart.js，色板与标记规范遵循 dataviz 参考实现）
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoK 1v1 训练看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
:root {
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
  --status-good: #0ca30c; --status-warning: #fab219;
  --status-serious: #ec835a; --status-critical: #d03b3b;
  --deem: #898781;
  color-scheme: light;
}
:root[data-theme="dark"] {
  --surface-1: #1a1a19; --page: #0d0d0d;
  --ink-1: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  color-scheme: dark;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #198f68;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--page); color: var(--ink-1);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--ink-2); font-size: 13px; margin-bottom: 16px; }
.filters { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.filters select, .filters button {
  font: inherit; font-size: 13px; padding: 6px 10px;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface-1); color: var(--ink-1);
}
.filters button { cursor: pointer; }
.filters button:hover { background: var(--grid); }
.filters label { font-size: 13px; color: var(--ink-2); }
.filters .chk { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
.filters input[type="range"] { width: 110px; vertical-align: middle; }
.mono { font-variant-numeric: tabular-nums; font-size: 12px; color: var(--ink-2); min-width: 34px; }
.kpis { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; min-width: 140px; flex: 1;
}
.tile .label { font-size: 12px; color: var(--ink-2); }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
.tile .note { font-size: 11px; color: var(--muted); margin-top: 2px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.card h2 { font-size: 14px; font-weight: 600; margin: 0 0 8px; color: var(--ink-1); }
.card .chart-box { position: relative; height: 220px; }
.card.mini .chart-box { height: 130px; }
.minis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.minis .card { padding: 10px 12px; }
.minis h2 { font-size: 12px; }
.minis .chart-box { height: 100px; }
#empty {
  display: none; padding: 40px; text-align: center; color: var(--ink-2);
  border: 1px dashed var(--border); border-radius: 8px;
}
table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  font-variant-numeric: tabular-nums; margin-top: 12px;
}
th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th { position: sticky; top: 0; background: var(--surface-1); color: var(--ink-2); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
#table-wrap { display: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-top: 16px; overflow-x: auto; }
#chartjs-fallback { display: none; color: var(--status-critical); margin-bottom: 12px; }
.legend-note { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<h1>HoK 1v1 训练看板</h1>
<div class="sub">数据源 logs/metrics.jsonl，每 5 秒自动刷新</div>
<div id="chartjs-fallback">Chart.js 加载失败（需要网络访问 CDN）。训练数据已保存，可稍后刷新重试。</div>
<div class="filters">
  <label>横轴范围</label>
  <select id="range">
    <option value="all">全部</option>
    <option value="5000">最近 5000 步</option>
    <option value="1000">最近 1000 步</option>
    <option value="200" selected>最近 200 步</option>
  </select>
  <label>数据</label>
  <select id="session">
    <option value="latest" selected>仅最新一次运行</option>
    <option value="all">全部运行（含历史）</option>
  </select>
  <label>平滑</label>
  <input type="range" id="smooth" min="0" max="95" value="60" step="5">
  <span id="smooth-val" class="mono">0.60</span>
  <label class="chk"><input type="checkbox" id="logy"> y 轴对数</label>
  <button id="theme-btn">切换深色模式</button>
  <button id="table-btn">显示数据表格</button>
</div>
<div id="empty">暂无指标数据。请先启动训练：python scripts/train_ppo.py</div>
<div id="kpis" class="kpis"></div>
<div class="grid">
  <div class="card"><h2>PPO 损失</h2><div class="chart-box"><canvas id="c_loss"></canvas></div></div>
  <div class="card"><h2>策略诊断</h2><div class="chart-box"><canvas id="c_diag"></canvas></div></div>
  <div class="card"><h2>胜率（训练 vs 评估）</h2><div class="chart-box"><canvas id="c_win"></canvas></div></div>
  <div class="card"><h2>学习率 / 熵系数（相对初始值）</h2><div class="chart-box"><canvas id="c_sched"></canvas></div></div>
  <div class="card"><h2>显存占用</h2><div class="chart-box"><canvas id="c_vram"></canvas></div></div>
  <div class="card"><h2>吞吐（FPS）</h2><div class="chart-box"><canvas id="c_fps"></canvas></div></div>
</div>
<h2 style="margin:20px 0 8px; font-size:14px;">奖励分量（每步均值）</h2>
<div id="reward-minis" class="minis"></div>
<h2 style="margin:20px 0 8px; font-size:14px;">动作分布（6 个头，前 3 高频动作 + 其他）</h2>
<div id="action-grid" class="grid"></div>
<div id="table-wrap"><h2>数据表格</h2><table id="tbl"></table></div>
<script>
"use strict";
const RE = { step: "step" };
const HEAD_NAMES = ["button(12)", "move_x", "move_z", "kill_x", "kill_z", "target(8)"];
const REWARD_NAMES = ["money", "exp", "hp", "ep", "kill", "dead", "tower", "last_hit", "time"];
const REWARD_LABELS = { money: "金钱", exp: "经验", hp: "血量", ep: "能量", kill: "击杀", dead: "死亡", tower: "推塔", last_hit: "补刀", time: "时间惩罚" };
const charts = {};
let records = [];
let rangeLimit = 200;

function pal() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    series: [v("--series-1"), v("--series-2"), v("--series-3"), v("--series-4"),
             v("--series-5"), v("--series-6"), v("--series-7"), v("--series-8")],
    ink1: v("--ink-1"), ink2: v("--ink-2"), muted: v("--muted"),
    grid: v("--grid"), axis: v("--axis"), surface: v("--surface-1"),
    warn: v("--status-warning"), serious: v("--status-serious"), critical: v("--status-critical"),
  };
}

function num(v) {
  if (v === undefined || v === null) return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function fmt(v) {
  if (!Number.isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a !== 0 && a < 1e-3) return v.toExponential(1);
  if (a < 1) return v.toFixed(3);
  if (a < 100) return v.toFixed(2);
  return v.toFixed(1);
}

//: 平滑系数（0 = 不平滑，0.9 = 强平滑）。对所有曲线统一生效。
let smoothing = 0.6;
//: y 轴是否用对数刻度（适合解释方差这类跨数量级的指标）。
let logScale = false;
//: 只看最近一个 session（训练重启会产生新 session），避免多次运行的数据混在一条曲线上。
let latestSessionOnly = true;

//: 把 x 轴主刻度算成**整数**步数。
//: 原先用线性轴的自动刻度，数据只有 3 个点时会给出 0 / 0.5 / 1.0 / 1.5 / 2.0
//: 这种"半步"刻度 —— 训练步数没有小数，读起来像时间轴又对不上。
function integerTicks(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return [min].filter(Number.isFinite);
  }
  const span = max - min;
  const steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
  let step = steps[steps.length - 1];
  for (const s of steps) {
    if (span / s <= 8) { step = s; break; }
  }
  const out = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + 1e-9; v += step) out.push(v);
  if (!out.length) out.push(Math.round(min));
  return out;
}

//: x 轴只显示整数步数（训练次数），不显示小数 / 时间。
function xScale(p) {
  return {
    type: "linear",
    grid: { color: p.grid },
    border: { color: p.axis },
    afterBuildTicks: (axis) => {
      axis.ticks = integerTicks(axis.min, axis.max).map((v) => ({ value: v }));
    },
    ticks: {
      color: p.muted, maxRotation: 0, autoSkip: false,
      callback: (v) => String(Math.round(v)),
    },
  };
}

function baseOpts() {
  const p = pal();
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: true, position: "top", align: "end",
        labels: { color: p.ink2, boxWidth: 10, boxHeight: 10, usePointStyle: true, padding: 10 },
      },
      //: 在右上角常驻显示"最后一个 step + 每条曲线的当前值"。
      //: 这是从 wandb 那类看板借来的做法，比把数值贴在曲线末端更清楚
      //: （末端贴字在曲线密集时会互相重叠、还会被画布边缘裁掉）。
      currentValues: { enabled: true, ink: p.ink1, muted: p.muted },
    },
    scales: {
      x: xScale(p),
      y: {
        type: logScale ? "logarithmic" : "linear",
        grid: { color: p.grid }, ticks: { color: p.muted },
        border: { color: p.axis },
      },
    },
  };
}

const currentValuesPlugin = {
  id: "currentValues",
  afterDatasetsDraw(chart, args, opts) {
    if (!opts || !opts.enabled) return;
    const sets = chart.data.datasets.filter((d) => d.data && d.data.length);
    if (!sets.length) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.font = "600 11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    const area = chart.chartArea;
    let y = area.top + 2;
    // 第一行：当前步数
    const lastPt = sets[0].data[sets[0].data.length - 1];
    ctx.fillStyle = opts.muted;
    ctx.fillText("step " + Math.round(lastPt.x), area.right - 2, y);
    y += 14;
    // 之后每行：曲线名 + 当前值（用曲线自己的颜色，方便对号）
    for (const ds of sets) {
      const v = ds.data[ds.data.length - 1].y;
      const text = (ds.label ? ds.label + " " : "") + fmt(v);
      const w = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(0,0,0,0)";
      ctx.fillStyle = ds.borderColor;
      ctx.fillText(text, area.right - 2, y);
      y += 13;
      if (y > area.bottom - 12) break;   // 行数过多就不画了，避免糊成一片
    }
    ctx.restore();
  },
};
Chart.register(currentValuesPlugin);

//: 指数滑动平均（EMA）。smoothing=0 时原样返回。
//: 训练曲线本身抖动很大（尤其 batch 只有几局时），平滑能让趋势看得出来，
//: 但**不要用它判断"某个点是否异常"** —— 需要看原始值时把滑块拉到 0。
function smoothData(data, alpha) {
  if (!alpha || alpha <= 0 || data.length < 3) return data;
  const out = [];
  let prev = NaN;
  for (const pt of data) {
    const y = Number(pt.y);
    if (!Number.isFinite(y)) { out.push({ x: pt.x, y: NaN }); continue; }
    prev = Number.isFinite(prev) ? alpha * prev + (1 - alpha) * y : y;
    out.push({ x: pt.x, y: prev });
  }
  return out;
}

function seriesXY(recs, key) {
  const data = recs.filter((r) => key in r).map((r) => ({ x: num(r.step), y: num(r[key]) }));
  return smoothData(data, smoothing);
}

function lineDS(label, recs, key, color, extra = {}) {
  return Object.assign({
    label, data: seriesXY(recs, key),
    borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
    pointHoverBorderWidth: 2, pointHoverBorderColor: pal().surface,
    tension: 0.25, spanGaps: true,
  }, extra);
}

function lineDSXY(label, data, color, extra = {}) {
  return Object.assign({
    label, data: smoothData(data, smoothing),
    borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
    pointHoverBorderWidth: 2, pointHoverBorderColor: pal().surface,
    tension: 0.25, spanGaps: true,
  }, extra);
}

function destroyGroup(prefix) {
  for (const k of Object.keys(charts)) {
    if (k.startsWith(prefix)) { charts[k].destroy(); delete charts[k]; }
  }
}

function makeLineChart(canvasId, datasets, optsExtra = {}) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const opts = baseOpts();
  Object.assign(opts, optsExtra);
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "line", data: { datasets }, options: opts,
  });
}

function renderKpis() {
  const el = document.getElementById("kpis");
  el.textContent = "";
  const last = records.length ? records[records.length - 1] : null;
  const win20 = records.slice(-20).map((r) => num(r["env/win_rate"])).filter(Number.isFinite);
  const len20 = records.slice(-20).map((r) => num(r["env/avg_episode_len"])).filter(Number.isFinite);
  const evalWins = records.filter((r) => "eval/win_rate" in r);
  const tiles = [
    { label: "当前 step", value: last ? String(last.step) : "-", note: "" },
    { label: "训练胜率（近 20 步）", value: win20.length ? (100 * win20.reduce((a, b) => a + b, 0) / win20.length).toFixed(0) + "%" : "-", note: "" },
    { label: "评估胜率", value: evalWins.length ? (100 * num(evalWins[evalWins.length - 1]["eval/win_rate"])).toFixed(0) + "%" : "-", note: "确定性动作" },
    { label: "平均对局时长（近 20 步）", value: len20.length ? (len20.reduce((a, b) => a + b, 0) / len20.length).toFixed(0) : "-", note: "决策步" },
    { label: "FPS", value: last ? fmt(num(last["res/fps"])) : "-", note: "" },
    { label: "显存", value: last ? fmt(num(last["res/vram_used_gb"])) + " GB" : "-", note: "已分配 / 已保留" },
  ];
  for (const t of tiles) {
    const d = document.createElement("div");
    d.className = "tile";
    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = t.label;
    const val = document.createElement("div");
    val.className = "value";
    val.textContent = t.value;
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = t.note;
    d.append(lab, val, note);
    el.appendChild(d);
  }
}

function renderCharts() {
  const p = pal();
  if (!records.length) {
    document.getElementById("empty").style.display = "block";
    return;
  }
  document.getElementById("empty").style.display = "none";

  makeLineChart("c_loss", [
    lineDS("policy_loss", records, "train/policy_loss", p.series[0]),
    lineDS("value_loss", records, "train/value_loss", p.series[1]),
  ]);

  makeLineChart("c_diag", [
    lineDS("kl", records, "train/kl", p.series[0]),
    lineDS("clip_fraction", records, "train/clip_fraction", p.series[1]),
    lineDS("explained_variance", records, "train/explained_variance", p.series[2]),
    lineDS("entropy", records, "train/entropy", p.series[3]),
  ]);

  makeLineChart("c_win", [
    lineDS("训练胜率", records, "env/win_rate", p.muted),
    lineDS("评估胜率", records, "eval/win_rate", p.series[0], { borderWidth: 3 }),
  ], {
    scales: {
      x: xScale(p),
      y: { min: 0, max: 1, grid: { color: p.grid }, ticks: { color: p.muted, callback: (v) => (100 * v).toFixed(0) + "%" }, border: { color: p.axis } },
    },
  });

  // 学习率 / 熵系数：按**各自第一个值**归一，方便一眼看出"是否在退火"。
  // ⚠️ 本项目的这两个值现在是**恒定**的（对齐官方：无 lr scheduler、β 恒 0.025），
  //    所以期望曲线就是贴着 1.0 的一条水平线。若看到它明显下滑，说明有人把
  //    退火重新打开了（旧配置会在 total_steps 内线性退到 0）。
  const lr0 = records.length ? num(records[0]["train/lr"]) : NaN;
  const beta0 = records.length ? num(records[0]["train/entropy_beta"]) : NaN;
  const schedDS = [
    lineDSXY("学习率（相对初始）", records.filter((r) => "train/lr" in r)
      .map((r) => ({ x: num(r.step), y: num(r["train/lr"]) / (lr0 || 1) })), p.series[0]),
    lineDSXY("熵系数（相对初始）", records.filter((r) => "train/entropy_beta" in r)
      .map((r) => ({ x: num(r.step), y: num(r["train/entropy_beta"]) / (beta0 || 1) })), p.series[1]),
  ];
  if (records.length) {
    const xs = [num(records[0].step), num(records[records.length - 1].step)];
    schedDS.push({
      label: "不退火（应为 1.0）",
      data: xs.map((x) => ({ x, y: 1.0 })),
      borderColor: p.muted, borderWidth: 1, borderDash: [4, 4],
      pointRadius: 0, tension: 0,
    });
  }
  makeLineChart("c_sched", schedDS, {
    scales: {
      x: xScale(p),
      y: { min: 0, max: 1.15, grid: { color: p.grid }, ticks: { color: p.muted }, border: { color: p.axis } },
    },
  });

  const oomG = records.filter((r) => "res/vram_oom_gb" in r);
  const oom = oomG.length ? num(oomG[0]["res/vram_oom_gb"]) : NaN;
  const deg = oomG.length ? num(oomG[0]["res/vram_degrade_gb"]) : NaN;
  const vramDS = [
    lineDS("已分配", records, "res/vram_used_gb", p.series[0]),
    lineDS("已保留", records, "res/vram_reserved_gb", p.series[1]),
  ];
  if (Number.isFinite(oom)) {
    vramDS.push({
      label: "OOM 阈值", data: records.map((r) => ({ x: num(r.step), y: oom })),
      borderColor: p.critical, borderWidth: 1, pointRadius: 0,
    });
  }
  if (Number.isFinite(deg)) {
    vramDS.push({
      label: "降级阈值", data: records.map((r) => ({ x: num(r.step), y: deg })),
      borderColor: p.warn, borderWidth: 1, pointRadius: 0,
    });
  }
  makeLineChart("c_vram", vramDS);

  makeLineChart("c_fps", [lineDS("FPS", records, "res/fps", p.series[0])]);
}

function renderRewardMinis() {
  const p = pal();
  destroyGroup("mini_");
  const host = document.getElementById("reward-minis");
  host.textContent = "";
  for (const name of REWARD_NAMES) {
    const card = document.createElement("div");
    card.className = "card mini";
    const h = document.createElement("h2");
    h.textContent = REWARD_LABELS[name];
    const box = document.createElement("div");
    box.className = "chart-box";
    const cv = document.createElement("canvas");
    box.appendChild(cv);
    card.append(h, box);
    host.appendChild(card);
    const key = "env/reward/" + name;
    const opts = baseOpts();
    // 单序列小卡：图例只会显示一个空标签，所以关掉它；
    // 数值由右上角的 currentValues 常驻显示。
    opts.plugins.legend.display = false;
    charts["mini_" + name] = new Chart(cv, {
      type: "line",
      data: { datasets: [lineDS("", records, key, p.series[0])] },
      options: opts,
    });
  }
}

function renderActionCharts() {
  const p = pal();
  destroyGroup("act_");
  const host = document.getElementById("action-grid");
  host.textContent = "";
  // 每个头的动作数不同（12/16/16/16/16/8），不能统一按 12 统计
  const HEAD_SIZES = [12, 16, 16, 16, 16, 8];
  for (let h = 0; h < 6; h++) {
    const card = document.createElement("div");
    card.className = "card mini";
    const head = document.createElement("h2");
    head.textContent = "head " + h + " " + HEAD_NAMES[h];
    const box = document.createElement("div");
    box.className = "chart-box";
    const cv = document.createElement("canvas");
    box.appendChild(cv);
    card.append(head, box);
    host.appendChild(card);

    const totals = new Map();
    for (const r of records) {
      for (let a = 0; a < HEAD_SIZES[h]; a++) {
        const k = "env/act_dist/head_" + h + "/" + a;
        if (k in r) totals.set(a, (totals.get(a) || 0) + num(r[k]));
      }
    }
    const sorted = Array.from(totals.entries()).sort((x, y) => y[1] - x[1]);
    const top = sorted.slice(0, 3).map((e) => e[0]);
    const datasets = [];
    top.forEach((a, i) => {
      datasets.push({
        label: "动作 " + a,
        data: records.map((r) => {
          const k = "env/act_dist/head_" + h + "/" + a;
          return { x: num(r.step), y: k in r ? num(r[k]) : NaN };
        }),
        borderColor: p.series[i], backgroundColor: hexA(p.series[i], 0.75),
        borderWidth: 2, pointRadius: 0, spanGaps: true, fill: true,
      });
    });
    datasets.push({
      label: "其他",
      data: records.map((r) => {
        let rest = 0;
        for (let a = 0; a < HEAD_SIZES[h]; a++) {
          if (top.includes(a)) continue;
          const k = "env/act_dist/head_" + h + "/" + a;
          if (k in r) rest += num(r[k]);
        }
        return { x: num(r.step), y: rest };
      }),
      borderColor: p.muted, backgroundColor: hexA(p.muted, 0.6),
      borderWidth: 2, pointRadius: 0, spanGaps: true, fill: true,
    });
    const opts = baseOpts();
    opts.plugins.legend = { display: true, labels: { color: p.ink2, boxWidth: 10, boxHeight: 10, usePointStyle: true } };
    opts.scales.x.stacked = true;
    opts.scales.y.stacked = true;
    charts["act_" + h] = new Chart(cv, {
      type: "line", data: { datasets }, options: opts,
    });
  }
}

function hexA(hex, alpha) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + alpha + ")";
}

function renderTable() {
  const tbl = document.getElementById("tbl");
  tbl.textContent = "";
  const keys = [];
  const seen = new Set();
  for (const r of records) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) { seen.add(k); keys.push(k); }
    }
  }
  keys.sort();
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");
  const thStep = document.createElement("th");
  thStep.textContent = "step";
  tr.appendChild(thStep);
  for (const k of keys.filter((k) => k !== "step")) {
    const th = document.createElement("th");
    th.textContent = k;
    tr.appendChild(th);
  }
  thead.appendChild(tr);
  tbl.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const r of records) {
    const row = document.createElement("tr");
    const td = document.createElement("td");
    td.textContent = String(r.step);
    row.appendChild(td);
    for (const k of keys.filter((k) => k !== "step")) {
      const c = document.createElement("td");
      c.textContent = k in r ? fmt(num(r[k])) : "-";
      row.appendChild(c);
    }
    tbody.appendChild(row);
  }
  tbl.appendChild(tbody);
}

function renderAll() {
  renderKpis();
  renderCharts();
  renderRewardMinis();
  renderActionCharts();
  renderTable();
}

//: 原始数据（含历史运行）与过滤后数据的分离。
//: ⚠️ 为什么需要它：每次启动训练都会往同一个 metrics.jsonl 追加，
//: 中间用 {"__session_end__": true} 分隔。若把多次运行画在同一条曲线上，
//: x 轴会出现"step 回到 0"的折返，看起来像指标炸了 —— 本看板此前就踩过这个。
let rawRecords = [];

function applyFilters() {
  records = latestSessionOnly ? latestSession(rawRecords) : rawRecords;
}

//: 取最近一个 session 的数据。
//:
//: metrics.jsonl 的结构是「若干条指标记录 + 一条 __session_end__ + 若干条 …」，
//: 每次启动训练都会追加。若把多次运行画在同一条曲线上，x 轴会出现"step 回到 0"
//: 的折返，看起来像指标炸了。
//:
//: 两种情况要照顾到：
//:   ① 最后一次运行还没有产生任何指标（刚写完分隔符就中断）→ 退回"全部运行"，
//:      否则看板会空白，让人误以为没数据；
//:   ② 无论走哪条分支都要**滤掉分隔符记录本身**，它只有 `__session_end__` 一个键，
//:      混进图表会变成一条全是 null 的曲线。
function latestSession(recs) {
  let endIdx = -1;                     // 最后一个分隔符在**原始数组**里的位置
  for (let i = recs.length - 1; i >= 0; i--) {
    if (recs[i] && recs[i].__session_end__) { endIdx = i; break; }
  }
  const clean = recs.filter((r) => r && !r.__session_end__);
  if (endIdx < 0) return clean;
  // 分隔符之前有多少条真实记录 —— 用计数而不是 indexOf，避免退化成 O(n²)。
  const before = recs.slice(0, endIdx).filter((r) => r && !r.__session_end__).length;
  const out = clean.slice(before);
  return out.length ? out : clean;    // 情况①：本次运行还没有数据 → 退回全部
}

//: 当前展示的是不是"仅最新一次运行"（用于 KPI 上的说明文字）。
function sessionNote() {
  if (!latestSessionOnly) return "全部运行";
  const n = rawRecords.filter((r) => r && !r.__session_end__).length;
  return records.length === n ? "全部运行（仅有一次）" : "最新一次运行";
}

async function load() {
  const limit = rangeLimit === "all" ? "" : "?limit=" + rangeLimit;
  try {
    const resp = await fetch("/api/metrics" + limit);
    rawRecords = await resp.json();
  } catch (e) {
    console.error("看板拉取数据失败:", e);
    rawRecords = [];
  }
  applyFilters();
  renderAll();
}

document.getElementById("range").addEventListener("change", (e) => {
  rangeLimit = e.target.value;
  load();
});
document.getElementById("session").addEventListener("change", (e) => {
  latestSessionOnly = e.target.value === "latest";
  applyFilters();
  renderAll();
});
document.getElementById("smooth").addEventListener("input", (e) => {
  smoothing = Number(e.target.value) / 100;
  document.getElementById("smooth-val").textContent = smoothing.toFixed(2);
  renderAll();
});
document.getElementById("logy").addEventListener("change", (e) => {
  logScale = e.target.checked;
  renderAll();
});
document.getElementById("theme-btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  renderAll();
});
document.getElementById("table-btn").addEventListener("click", () => {
  const wrap = document.getElementById("table-wrap");
  wrap.style.display = wrap.style.display === "block" ? "none" : "block";
});

window.addEventListener("load", () => {
  if (typeof Chart === "undefined") {
    document.getElementById("chartjs-fallback").style.display = "block";
  }
  load();
  setInterval(load, 5000);
});
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """看板 HTTP 服务：GET / 返回页面，GET /api/metrics 返回指标 JSON。"""

    store: MetricsStore = None

    def do_GET(self):      # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            qs = parse_qs(parsed.query)
            limit = None
            if "limit" in qs and qs["limit"] and qs["limit"][0].isdigit():
                limit = int(qs["limit"][0])
            body = json.dumps(self.store.tail(limit)).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif parsed.path in ("/", "/index.html"):
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if ctype.startswith("application/json"):
            # 禁止缓存：训练数据每步都在变，缓存会让看板停在旧数据
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):     # noqa: A002
        pass


def main():
    parser = argparse.ArgumentParser(description="HoK 1v1 训练实时看板")
    parser.add_argument("--port", type=int, default=8600, help="监听端口（默认 8600）")
    parser.add_argument(
        "--metrics",
        default=str(_HERE.parent / "logs" / "metrics.jsonl"),
        help="metrics.jsonl 路径（默认 <项目根>/logs/metrics.jsonl）",
    )
    args = parser.parse_args()

    DashboardHandler.store = MetricsStore(args.metrics, read_only=True)
    # 绑定 0.0.0.0：WSL2 下 Windows 浏览器经 localhost 访问镜像回环，
    # 绑 127.0.0.1 时部分 WSL 网络状态会让 Windows 侧连不上（实测镜像
    # 模式下 0.0.0.0 绑定对 Windows localhost 可达）
    # allow_reuse_address：重启时跳过 TIME_WAIT，避免端口短暂不可用
    DashboardHandler.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    except OSError as exc:
        if exc.errno == 98:      # EADDRINUSE
            print(f"端口 {args.port} 已被占用，通常原因：")
            print("  1. 已经有一个看板在运行（直接刷新浏览器页面即可，不用重复启动）")
            print("  2. 有残留进程占着端口，用下面命令找到并清理：")
            print(f"     ss -tlnp | grep {args.port}")
            print("     kill <pid>")
            sys.exit(1)
        raise
    print(f"看板已启动: http://localhost:{args.port} （指标文件: {args.metrics}）")
    print("训练过程中浏览器页面每 5 秒自动刷新。Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n看板已退出。")
        server.server_close()


if __name__ == "__main__":
    main()
