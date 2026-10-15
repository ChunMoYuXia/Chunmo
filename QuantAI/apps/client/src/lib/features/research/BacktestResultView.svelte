<script lang="ts">
  import { resolve } from '$app/paths';
  import type { components } from '$lib/api/schema';
  import { number } from '$lib/utils/format';
  import MetricCard from './MetricCard.svelte';
  import PerformanceChart from './PerformanceChart.svelte';
  let { artifact }: { artifact: components['schemas']['ResearchArtifact'] } = $props();
  let tab = $state<'trades' | 'unfilled' | 'assumptions'>('trades');
  let limit = $state(50);
</script>

<section class="panel">
  <div class="section-heading">
    <h2>回测结果</h2>
    <a href={resolve(`/reports?research=${encodeURIComponent(artifact.id)}`)}>查看证据报告 →</a>
    <span class:synthetic={artifact.snapshot.source_kind === 'synthetic'} class="badge"
      >{artifact.snapshot.source_kind === 'synthetic' ? '合成演示' : '本地导入'} · v{artifact
        .snapshot.version}</span
    >
  </div>
  <div class="stats-row">
    <MetricCard label="区间收益" metric={artifact.metrics.total_return} /><MetricCard
      label="最大回撤"
      metric={artifact.metrics.max_drawdown}
    /><MetricCard label="年化波动" metric={artifact.metrics.annualized_volatility} /><MetricCard
      label="夏普比率"
      metric={artifact.metrics.sharpe}
    />
  </div>
  <PerformanceChart result={artifact.result} />
  <div class="stats-row">
    <MetricCard label="年化收益" metric={artifact.metrics.annualized_return} /><MetricCard
      label="基准收益"
      metric={artifact.metrics.benchmark_return}
    /><MetricCard label="超额收益" metric={artifact.metrics.excess_return} />
    <div class="stat">
      <small>显式费用 / 元</small><strong>{number(artifact.result.costs)}</strong><small
        >滑点价差 {number(artifact.result.slippage_costs)} 元，已计入成交价</small
      >
    </div>
  </div>
  <p class="context-note">
    {artifact.result.request.start} 至 {artifact.result.request.end} · 年化 {artifact.metrics
      .annualization} 日 · 数据 {artifact.snapshot.dataset_id} · 引擎 {artifact.result.evidence
      .engine_version}
  </p>
  <details>
    <summary>数据与计算版本</summary><code class="hash">{artifact.snapshot.content_hash}</code>
    <p class="context-note">
      {artifact.snapshot.calendar_id} · {artifact.result.evidence.price_basis} · 费率 {artifact.result.evidence.fee_versions.join(
        '、',
      )} · {artifact.result.evidence.knowledge_mode}
    </p>
  </details>
</section>
<section class="panel">
  <div class="actions">
    <button aria-pressed={tab === 'trades'} onclick={() => (tab = 'trades')}>成交记录</button
    ><button aria-pressed={tab === 'unfilled'} onclick={() => (tab = 'unfilled')}>未成交</button
    ><button aria-pressed={tab === 'assumptions'} onclick={() => (tab = 'assumptions')}
      >假设与缺口</button
    >
  </div>
  {#if tab === 'trades'}<div class="table-scroll">
      <table>
        <thead
          ><tr
            ><th>交易日</th><th>证券</th><th>方向</th><th>数量</th><th>成交价 / 元</th><th
              >佣金 / 元</th
            ><th>过户费 / 元</th><th>印花税 / 元</th></tr
          ></thead
        ><tbody
          >{#each artifact.result.trades.slice(0, limit) as trade (trade.order_id)}<tr
              ><td>{trade.fill.session}</td><td>{trade.fill.security_id}</td><td
                >{trade.fill.side === 'buy' ? '买入' : '卖出'}</td
              ><td>{trade.fill.quantity}</td><td>{number(trade.fill.price)}</td><td
                >{number(trade.fees.commission)}</td
              ><td>{number(trade.fees.transfer)}</td><td>{number(trade.fees.stamp)}</td></tr
            >{/each}</tbody
        >
      </table>
    </div>
    <p class="context-note">共 {artifact.result.trades.length} 笔成交。</p>
    {#if artifact.result.trades.length > limit}<button onclick={() => (limit += 100)}
        >显示更多成交</button
      >{/if}
  {:else if tab === 'unfilled'}<div class="table-scroll">
      <table>
        <thead><tr><th>证券</th><th>状态</th><th>原因</th></tr></thead><tbody
          >{#each artifact.result.unfilled.slice(0, limit) as item (item.order.order_id)}<tr
              ><td>{item.order.security_id}</td><td>{item.order.state}</td><td>{item.reason}</td
              ></tr
            >{/each}</tbody
        >
      </table>
    </div>
    {#if artifact.result.unfilled.length > limit}<button onclick={() => (limit += 100)}
        >显示更多未成交</button
      >{/if}
  {:else}{#each [...artifact.result.assumptions, ...artifact.result.warnings, ...artifact.metrics.warnings] as note, index (index)}<p
        class="context-note"
      >
        {note}
      </p>{/each}{/if}
</section>
