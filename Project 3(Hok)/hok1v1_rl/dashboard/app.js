/* 页面只读取日志。SVG 绘图不依赖 Chart.js、外部字体或网络。 */
'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = x => typeof x === 'number' && Number.isFinite(x);
const number = (v, digits = 2) => finite(v) ? v.toLocaleString('zh-CN', {maximumFractionDigits: digits}) : '—';
const percent = v => finite(v) ? `${(v * 100).toFixed(1)}%` : '—';
const compact = v => !finite(v) ? '—' : Math.abs(v) >= 1e6 ? `${(v/1e6).toFixed(1)}m` : Math.abs(v) >= 1e4 ? `${(v/1e3).toFixed(1)}k` : Math.abs(v) > 0 && Math.abs(v) < .001 ? v.toExponential(1) : number(v, 3);
function episodeTime(game) {
  const value=game.started_at || game.timestamp, date=new Date(value);
  if (!value || !Number.isFinite(date.getTime())) return '—';
  const prefix=game.started_at_estimated ? '≈ ' : game.started_at ? '' : '结束 ';
  const note=game.started_at_estimated ? '旧日志按结束时间与轨迹耗时估算开局' : game.started_at ? '发起对局的时间' : '旧日志仅有结束写入时间';
  return `<span title="${escapeHTML(`${note}；原始时间 ${value}`)}">${prefix}${escapeHTML(date.toLocaleString('zh-CN',{hour12:false}))}</span>`;
}
const state = {data: null, run: '', paused: false, alpha: .6, range: 200, axis: 'step', modal: null, loading: false};
const colors = ['var(--accent)', 'var(--blue)', 'var(--orange)'];
const charts = [
  {id:'wins', area:'overview', title:'获胜趋势', note:'训练为每批对局胜率；快评为固定规则评估。虚线是 90% 目标。', keys:['env/win_rate','eval/win_rate'], names:['训练采样','评估'], pct:true, reference:.9, domain:[0,1], noSmooth:true},
  {id:'reward', area:'overview', title:'每局总回报', note:'先看长期趋势，再看奖励分量是否与获胜一致。', keys:['env/avg_reward'], names:['总回报']},
  {id:'policy', title:'Policy loss · 策略损失', note:'优化目标随采样策略变化；数值正负不直接表示胜负。', keys:['train/policy_loss']},
  {id:'value', title:'Value loss · 价值损失', note:'价值预测与 GAE 回报目标的误差，结合 EV 一起看。', keys:['train/value_loss']},
  {id:'kl', title:'KL · 新旧策略差异', note:'过大可能更新激进；过小也可能是更新量或优势信号弱。', keys:['train/kl']},
  {id:'clip', title:'Clip fraction · 裁剪比例', note:'进入 PPO 裁剪区的有效动作头比例。', keys:['train/clip_fraction'], pct:true, domain:[0,1]},
  {id:'entropy', title:'Entropy · 探索程度', note:'有效动作头的平均熵；合法动作数量也会影响它。', keys:['train/entropy']},
  {id:'ev', title:'Explained variance · 价值解释度', note:'越接近 1，价值预测解释回报波动的能力越好。', keys:['train/explained_variance'], reference:0},
  {id:'grad', title:'Gradient norm · 梯度范数', note:'记录裁剪前的梯度范数；非有限数值按缺失处理。', keys:['train/grad_norm']},
  {id:'actor-critic-grad', title:'策略与价值梯度', note:'更新前第一批序列；策略含熵项，价值含 value_coef。比较两者尺度。', keys:['train/policy_grad_norm','train/value_grad_norm'], names:['策略','价值']},
  {id:'gradient-clip', title:'动作头梯度保留比例', note:'实线记录动作头实际裁剪比例；另一条为首批序列的全局裁剪估计，样本口径不同。', keys:['train/policy_head_clip_scale','train/estimated_global_clip_scale'], names:['实际动作头 · 更新均值','假设全局裁剪 · 首批'], pct:true, domain:[0,1], noSmooth:true},
  {id:'duration', title:'每次更新耗时 · 秒', note:'采样、PPO 更新与快评分开记录；未快评时评估耗时为 0。', keys:['res/sampling_seconds','res/update_seconds','res/evaluation_seconds'], names:['采样','PPO 更新','快评'], noSmooth:true},
  {id:'fountain', title:'泉水停留与技能尝试', note:'泉水中心 6000 单位内的近似区域。停留按坐标已知的存活决策计；技能按坐标已知的技能尝试计，不代表成功施法。', keys:['behavior/fountain_rate','behavior/fountain_skill_attempt_rate'], names:['泉水停留比例','技能尝试在泉水的比例'], pct:true, domain:[0,1], noSmooth:true},
  {id:'lr', title:'Learning rate · 学习率', note:'对照实际学习率变化，判断策略更新量。', keys:['train/lr'], noSmooth:true},
  {id:'fps', title:'吞吐 · 决策 / 秒', note:'采样与更新期间的真实决策吞吐，不包含评估耗时。', keys:['res/fps']},
  {id:'length', title:'对局长度 · 决策数', note:'每局真实决策均值，不包含 LSTM padding。', keys:['env/avg_episode_len']},
  {id:'invalid', title:'未知与截断比例', note:'未知结局与外部截断可能重叠；先检查环境与终局逻辑。', keys:['env/unknown_rate','env/truncated_rate'], names:['未知','截断'], pct:true, domain:[0,1], noSmooth:true},
  {id:'vram', title:'GPU 显存 · GiB', note:'PyTorch 已分配 / 已保留显存，不含其他进程占用。', keys:['res/vram_used_gb','res/vram_reserved_gb'], names:['已分配','已保留']}
];

function createCards() {
  for (const spec of charts) {
    const card = document.createElement('article');
    card.className = 'chart-card'; card.tabIndex = 0; card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `放大${spec.title}`); card.id = `card-${spec.id}`;
    card.innerHTML = `<div class="chart-title"><strong>${spec.title}</strong><span class="chart-value"></span></div><p class="chart-description">${spec.note}</p><div class="chart-body"></div>`;
    card.onclick = () => openChart(spec);
    card.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); openChart(spec);} };
    $(spec.area === 'overview' ? 'overview-charts' : 'metric-charts').append(card);
  }
}

function seriesFor(spec) {
  const rows = (state.data?.metrics || []).slice(-state.range);
  return spec.keys.map((key, i) => {
    let average = null;
    const points = rows.map(row => {
      const raw = row[key], x = row[state.axis];
      if (!finite(raw) || !finite(x)) return null;
      average = average === null || spec.noSmooth ? raw : state.alpha * average + (1 - state.alpha) * raw;
      return {x, raw, y:average, step:row.step};
    });
    return {name:spec.names?.[i] || spec.title.split(' · ')[0], color:colors[i], points};
  });
}

function drawChart(container, spec, large = false) {
  const series = seriesFor(spec), points = series.flatMap(s => s.points.filter(Boolean));
  if (!points.length) {container.innerHTML = '<div class="empty-chart">等待该指标 · 缺失数据不会补成 0</div>'; return;}
  const width = innerWidth <= 700 ? 360 : 640;
  const height = large ? 360 : 230, left = 56, right = 15, top = 15, bottom = 31;
  let xMin = Math.min(...points.map(p => p.x)), xMax = Math.max(...points.map(p => p.x));
  if (xMin === xMax) {xMin -= .5; xMax += .5;}
  let low = spec.domain?.[0] ?? Math.min(...points.map(p => Math.min(p.y, p.raw)), spec.reference ?? Infinity);
  let high = spec.domain?.[1] ?? Math.max(...points.map(p => Math.max(p.y, p.raw)), spec.reference ?? -Infinity);
  if (!spec.domain) {const padding = (high - low || Math.abs(high) * .1 || 1) * .12; low -= padding; high += padding;}
  const px = x => left + (x - xMin) / (xMax - xMin) * (width - left - right);
  const py = y => top + (high - y) / (high - low) * (height - top - bottom);
  const valueLabel = v => spec.pct ? `${Math.round(v*100)}%` : compact(v);
  let svg = `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(spec.title)}随${state.axis === 'step' ? '更新' : '真实决策数'}变化">`;
  for (let i=0; i<=4; i++) {
    const y = low + (high-low)*i/4, at = py(y);
    svg += `<line class="grid" x1="${left}" y1="${at}" x2="${width-right}" y2="${at}"/><text x="${left-9}" y="${at+3}" text-anchor="end">${valueLabel(y)}</text>`;
    const x = xMin + (xMax-xMin)*i/4;
    svg += `<text x="${px(x)}" y="${height-8}" text-anchor="middle">${compact(x)}</text>`;
  }
  if (finite(spec.reference)) svg += `<line class="reference" x1="${left}" y1="${py(spec.reference)}" x2="${width-right}" y2="${py(spec.reference)}"/>`;
  function pathFor(data, field, connect) {
    let path = '', started = false;
    for (const p of data) {
      if (!p) {if (!connect) started=false; continue;}
      path += `${started ? 'L':'M'}${px(p.x).toFixed(2)},${py(p[field]).toFixed(2)} `; started=true;
    }
    return path;
  }
  for (const s of series) {
    const available = s.points.filter(Boolean);
    svg += `<path class="raw" d="${pathFor(s.points,'raw',spec.noSmooth)}" fill="none" stroke="${s.color}" stroke-width="1.3"/>`;
    svg += `<path d="${pathFor(s.points,'y',spec.noSmooth)}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
    for (const p of available) {
      if (spec.noSmooth || available.length === 1) svg += `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="3" fill="${s.color}"/>`;
    }
  }
  svg += '</svg><div class="legend">' + series.map(s => `<span><i style="background:${s.color}"></i>${escapeHTML(s.name)}</span>`).join('') + `<span>${spec.noSmooth ? '原始值' : `浅线原始值 · 深线 EMA ${state.alpha.toFixed(2)}`}</span></div>`;
  container.innerHTML = svg;
  const element = container.querySelector('svg');
  element.onpointermove = event => {
    const rect = element.getBoundingClientRect();
    const x = xMin + ((event.clientX-rect.left)/rect.width*width-left)/(width-left-right)*(xMax-xMin);
    const lines = series.map(s => {
      const available = s.points.filter(Boolean);
      if (!available.length) return null;
      const p = available.reduce((a,b) => Math.abs(b.x-x) < Math.abs(a.x-x) ? b:a);
      return `${s.name}  ·  update ${p.step}\n原始 ${spec.pct ? percent(p.raw):number(p.raw,6)}${spec.noSmooth ? '':`  /  平滑 ${number(p.y,6)}`}`;
    }).filter(Boolean);
    $('tooltip').textContent = lines.join('\n'); $('tooltip').hidden = false;
    $('tooltip').style.left = `${Math.max(8,Math.min(event.clientX+12,innerWidth-$('tooltip').offsetWidth-12))}px`;
    $('tooltip').style.top = `${Math.max(8,Math.min(event.clientY+12,innerHeight-$('tooltip').offsetHeight-12))}px`;
  };
  element.onpointerleave = () => {$('tooltip').hidden = true;};
}

function openChart(spec) {
  state.modal = spec; $('tooltip').hidden = true;
  $('modal-title').textContent = spec.title; $('modal-note').textContent = spec.note;
  if (!$('chart-modal').open) $('chart-modal').showModal();
  drawChart($('modal-chart'), spec, true);
}

function renderCharts() {
  for (const spec of charts) {
    const card = $(`card-${spec.id}`), series = seriesFor(spec);
    const latest = series[0].points.filter(Boolean).at(-1);
    card.querySelector('.chart-value').textContent = spec.pct ? percent(latest?.raw) : compact(latest?.raw);
    drawChart(card.querySelector('.chart-body'), spec);
  }
  if (state.modal && $('chart-modal').open) drawChart($('modal-chart'), state.modal, true);
}

function renderSummary() {
  const d=state.data, rows=d.metrics || [], last=rows.at(-1) || {}, status=d.status || {}, manifest=d.manifest || {};
  const completed=status.completed_updates ?? last.step ?? 0, target=status.target_updates ?? manifest.args?.total_steps;
  $('run-name').textContent=d.run?.name || '等待训练日志';
  $('update-count').textContent=`${number(completed,0)} / ${number(target,0)}`;
  $('progress').value=finite(target) && target>0 ? Math.min(100,completed/target*100):0;
  const phases={starting:'准备环境',sampling:'收集训练对局',updating:'PPO 策略更新',evaluating:'固定规则评估',saving:'保存模型',finished:'训练已完成',interrupted:'训练已中断，正在保存或已保存',failed:'训练异常退出'};
  const age=status.heartbeat ? Math.max(0,(Date.now()-Date.parse(status.heartbeat))/1000):null;
  $('phase').textContent=status.phase ? `${phases[status.phase] || status.phase}${age>30 && !['finished','interrupted','failed'].includes(status.phase) ? ' · 状态超过 30 秒未更新，可能正在等待环境响应':''}`:'日志模式 · 局内进度从下次启动新版训练时生效';
  $('live-detail').textContent=status.hero_id ? `当前对局：${status.hero_id} vs ${status.opponent_id} · ${status.player_id===0?'蓝方':'红方'} · ${number(status.episode_decisions,0)} 决策${status.phase==='evaluating' ? ` · 评估 ${number(status.eval_completed,0)} / ${number(status.eval_target,0)} 局`:''}` : `已记录 ${number(d.train?.games,0)} 局训练 · 真实决策 ${number(last['env/total_env_steps'],0)}`;
  if (status.sampler_count > 1) {
    const workers=Object.entries(status.workers || {});
    const detail=workers.map(([id,w])=>`采样器 ${Number(id)+1}：${w.hero_id ?? '—'} / ${w.opponent_id ?? '—'} · ${w.player_id===0?'蓝方':'红方'} · ${number(w.episode_decisions,0)} 决策${w.phase==='idle'?' · 已收齐':''}`).join('；');
    $('live-detail').textContent=`${status.sampler_count} 个采样进程${detail ? ` · ${detail}`:''}${status.phase==='evaluating' ? ` · 评估 ${number(status.eval_completed,0)} / ${number(status.eval_target,0)} 局`:''}`;
  }
  const modified = status.heartbeat ? new Date(status.heartbeat) : d.run ? new Date(d.run.modified*1000):null;
  $('updated').textContent=modified ? `数据更新 ${modified.toLocaleString('zh-CN',{hour12:false})}`:'等待首次写入';
  const ev=d.evaluation;
  const kpis=[['最近训练胜率',percent(d.recent?.win_rate),`最近窗口内 ${number(d.recent?.games,0)} 局 · 按局统计`],['最近快评胜率',percent(ev?.win_rate),ev ? `${number(ev.games,0)} 局 · update ${ev.update}`:'尚未到达评估间隔'],['每局回报',number(last['env/avg_reward']), '最近一次 PPO 更新'],['真实决策数',compact(last['env/total_env_steps']),'累计采样 · 不含 padding'],['英雄已覆盖',number(Object.keys(d.heroes?.train || {}).length,0),'当前运行 · 已完成训练对局'],['决策 / 秒',number(last['res/fps'],1),'采样 + 更新 · 不含评估']];
  $('kpis').innerHTML=kpis.map(([label,value,note])=>`<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${escapeHTML(value)}</div><div class="kpi-note">${escapeHTML(note)}</div></div>`).join('');
  const formal=d.independent_evaluation, summary=formal || ev;
  $('goal-title').textContent=formal?.goal_verified ? '独立评估通过目标' : formal ? '已有独立评估，尚未通过目标' : ev ? '已有快评，等待独立验证':'等待首次评估';
  $('goal-text').textContent=summary ? `${formal?'独立评估':'训练快评'}胜率 ${percent(summary.win_rate)}，95% 区间 ${percent(summary.win_rate_ci95?.[0])} – ${percent(summary.win_rate_ci95?.[1])}。${formal ? '是否通过以评估的局数、区间下界与异常结果共同判断。':'快评用于观察趋势，不作为正式达标结论。'}` : '90% 是训练目标。训练采样胜率不会被当作已达标的证明。';
  $('evaluation-detail').innerHTML=summary ? `<span>${number(summary.games,0)} 局</span><span>未知 ${number(summary.unknown,0)}</span><span>截断 ${number(summary.truncated,0)}</span><span>目标 ${percent(summary.goal ?? .9)}</span>`:'';
  const insights=[];
  if (!rows.length) insights.push('等待首次 PPO 更新；采样对局完成后才能生成这一批学习指标。');
  if (!status.heartbeat) insights.push('当前运行没有局内遥测，页面会在对局或更新完成后变化。');
  if (!ev) insights.push(`首次快评尚未记录${manifest.config?.train?.eval_interval_steps ? `，每 ${manifest.config.train.eval_interval_steps} 次更新评估一次`:''}。`);
  if (d.train?.unknown || d.train?.truncated) insights.push(`注意：训练中有 ${d.train.unknown || 0} 局未知、${d.train.truncated || 0} 局截断，应先检查环境与终局逻辑。`);
  if (finite(last['train/explained_variance']) && last['train/explained_variance']<0) insights.push('价值解释度低于 0：当前估值较弱，需要结合回报波动与后续趋势观察。');
  insights.push('随机英雄任务请检查英雄覆盖与双方阵营表现，整体平均值可能掩盖短板。');
  if (status.error) insights.push(`训练异常：${status.error}`);
  $('insights').innerHTML=insights.map(s=>`<li>${escapeHTML(s)}</li>`).join('');
  $('run-config').textContent=`运行：${d.run?.id || '—'}；创建时间：${manifest.created_at || '旧日志未记录'}；Python ${manifest.python || '—'} / PyTorch ${manifest.torch || '—'}。`;
}

function renderBars() {
  const last=state.data.metrics?.at(-1) || {};
  function bars(id, entries, asPercent=false) {
    const max=Math.max(...entries.map(([,v])=>Math.abs(v)),1e-9);
    $(id).innerHTML=entries.length ? entries.map(([label,value])=>`<div class="bar-row ${value<0?'negative':''}"><span>${escapeHTML(label)}</span><span class="bar-track"><i style="width:${Math.abs(value)/max*100}%"></i></span><span class="bar-value">${asPercent ? percent(value):number(value,3)}</span></div>`).join(''):'<p class="caption">该运行尚无此类指标</p>';
  }
  bars('rewards',Object.entries(last).filter(([k,v])=>k.startsWith('env/reward/') && finite(v)).map(([k,v])=>[k.split('/').at(-1),v]));
  const prefix=`act_dist/head_${$('action-head').value}/`;
  bars('actions',Object.entries(last).filter(([k,v])=>k.startsWith(prefix) && finite(v)).map(([k,v])=>[k.split('/').at(-1),v]),true);
}

function renderHeroes() {
  const d=state.data, mode=$('hero-mode').value, latest=mode==='formal' ? d.independent_evaluation : d.evaluation;
  // 训练累计表与最近一次完整快评表不能混为同一批样本。
  const groups=mode==='train' ? d.heroes?.train || {} : latest?.by_hero || {};
  const sides=mode==='train' ? d.sides?.train || {} : latest?.by_side || {};
  const expected=d.manifest?.config?.env?.hero_ids || [];
  const ids=[...new Set([...Object.keys(groups),...expected.map(String)])].sort((a,b)=>Number(a)-Number(b));
  $('hero-note').textContent=mode==='train' ? '当前运行的累计训练对局。平局与未知计入分母；零样本英雄不会显示为 0% 胜率。':`${mode==='formal'?'最近一次独立评估':'最近一次完整快评'}，按己方英雄与阵营分组。各英雄样本有限，整体目标通过不证明每个英雄都达标。未记录的细项显示“—”。`;
  $('sides').innerHTML=[0,1].map(side=>{const s=sides[String(side)];return `<div class="panel"><div><h3>${side===0?'蓝方 · player 0':'红方 · player 1'}</h3><span class="muted">${number(s?.games,0)} 局</span></div><strong>${percent(s?.win_rate)}</strong></div>`;}).join('');
  $('hero-table').innerHTML=ids.length ? ids.map(id=>{const s=groups[id] || {}, rate=s.win_rate; return `<tr><td>${escapeHTML(id)}</td><td>${number(s.games ?? 0,0)}</td><td>${number(s.win ?? s.wins,0)} / ${number(s.lose,0)}</td><td>${number(s.draw,0)} / ${number(s.unknown,0)} / ${number(s.truncated,0)}</td><td><span class="win-meter">${percent(rate)}<i style="width:${finite(rate)?rate*70:0}px"></i></span></td><td class="muted">${!s.games?'尚无样本':s.games<30?'样本少，先看覆盖':'结合区间与评估趋势'}</td></tr>`;}).join(''):'<tr><td colspan="6" class="muted">尚无分英雄对局数据；旧 metrics.jsonl 不包含逐局记录。</td></tr>';
  const outcomes={win:'胜利',lose:'失败',draw:'平局',unknown:'未知'};
  $('episode-table').innerHTML=[...(d.episodes || [])].reverse().map(g=>{
    const control=g.control, hero=control?.hero_id ?? g.hero_id, opponent=control?.opponent_id ?? g.opponent_id;
    const side=control?.camp ? control.camp-1 : g.player_id;
    const identity=control?.verified ? `已校验 · runtime_player_id=${control.runtime_player_id}` : '旧日志未记录控制方校验';
    return `<tr><td>${episodeTime(g)}</td><td>${number(g.update,0)}</td><td>${g.mode==='train'?'训练':'评估'}</td><td title="${escapeHTML(identity)}">${escapeHTML(hero)} / ${escapeHTML(opponent)}</td><td title="${escapeHTML(identity)}">${side===0?'蓝方':side===1?'红方':'—'}</td><td><span class="tag ${escapeHTML(g.outcome)}">${g.truncated?'截断':escapeHTML(outcomes[g.outcome] || g.outcome)}</span></td><td>${number(g.episode_len,0)}</td><td>${number(g.reward)}</td><td>${escapeHTML(g.sdk_client_id || g.sgame_id || g.game_id || '—')}</td></tr>`;
  }).join('') || '<tr><td colspan="9" class="muted">等待逐局日志</td></tr>';
}

function renderButtons() {
  const last=state.data.metrics?.at(-1) || {}, mode=$('button-mode').value;
  const diagnostic=state.data.evaluation?.policy_diagnostics || {};
  const names=['None1','None2','Move','Attack','Skill1','Skill2','Skill3','HealSkill','ChosenSkill','Recall','Skill4','EquipSkill'];
  const fields=['legal_rate','probability_mean','selection_rate','greedy_rate'];
  const rows=names.map((name,i)=>{
    const values=fields.map(key=>mode==='eval' ? diagnostic[key]?.[i] : last[`policy/button_${key}/${i}`]);
    return `<tr><td>${i} · ${name}</td>${values.map(value=>`<td>${percent(value)}</td>`).join('')}</tr>`;
  });
  $('button-table').innerHTML=rows.join('');
}

let chartVersion = '';
function render() {
  renderSummary(); renderBars(); renderHeroes(); renderButtons();
  const version = JSON.stringify([state.data.run?.id, state.data.metrics]);
  if (version !== chartVersion) {renderCharts(); chartVersion = version;}
}
async function load() {
  if (state.paused || state.loading) return;
  state.loading=true;
  try {
    const response=await fetch(`/api/state?limit=5000${state.run ? `&run=${encodeURIComponent(state.run)}`:''}`,{cache:'no-store',signal:AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data=await response.json();
    $('run').innerHTML=(state.data.runs || []).map(r=>`<option value="${escapeHTML(r.id)}">${escapeHTML(r.name)}</option>`).join('') || '<option value="">等待运行目录</option>';
    if (state.data.run) {state.run=state.data.run.id; $('run').value=state.run;}
    $('error').hidden=true; $('connection').textContent='● 实时读取'; render();
  } catch(error) {
    $('error').textContent=`连接暂时中断：${error.message}。保留最后一次数据，稍后自动重试。`;
    $('error').hidden=false; $('connection').textContent='○ 连接中断';
  } finally {state.loading=false;}
}

createCards();
try {document.documentElement.dataset.theme=localStorage.getItem('hok-theme') || 'light';} catch (_) {}
$('theme').onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark'; document.documentElement.dataset.theme=theme; try{localStorage.setItem('hok-theme',theme);}catch(_){};};
for (const button of document.querySelectorAll('[data-tab]')) button.onclick=()=>{
  for (const b of document.querySelectorAll('[data-tab]')) {b.classList.toggle('active',b===button); $(b.dataset.tab).hidden=b!==button;}
};
$('run').onchange=()=>{state.run=$('run').value; load();};
$('range').onchange=()=>{state.range=Number($('range').value); if(state.data)renderCharts();};
$('axis').onchange=()=>{state.axis=$('axis').value; if(state.data)renderCharts();};
$('smooth').oninput=()=>{state.alpha=Number($('smooth').value); $('smooth-value').textContent=state.alpha.toFixed(2); if(state.data)renderCharts();};
$('hero-mode').onchange=()=>{if(state.data)renderHeroes();};
$('action-head').onchange=()=>{if(state.data)renderBars();};
$('button-mode').onchange=()=>{if(state.data)renderButtons();};
$('pause').onclick=()=>{state.paused=!state.paused; $('pause').textContent=state.paused?'恢复刷新':'暂停刷新'; $('connection').textContent=state.paused?'○ 已暂停刷新':'● 实时读取'; if(!state.paused)load();};
$('close-modal').onclick=()=>{$('chart-modal').close(); $('tooltip').hidden=true; state.modal=null;};
$('chart-modal').addEventListener('close',()=>{$('tooltip').hidden=true; state.modal=null;});
$('chart-modal').onclick=event=>{if(event.target===$('chart-modal')) {const rect=$('chart-modal').getBoundingClientRect(); if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)$('close-modal').click();}};
$('download').onclick=()=>{if(!state.data)return; const url=URL.createObjectURL(new Blob([JSON.stringify(state.data,null,2)],{type:'application/json'})); const a=document.createElement('a'); a.href=url; a.download=`hok-${state.data.run?.name || 'run'}-dashboard.json`; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);};
load(); setInterval(load,2000);
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer); resizeTimer=setTimeout(()=>{if(state.data)renderCharts();},150);});
