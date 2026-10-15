<script lang="ts">
  import type { components } from '$lib/api/schema';
  import { number, percent } from '$lib/utils/format';
  import MetricCard from './MetricCard.svelte';
  let { risk }: { risk: components['schemas']['PortfolioRiskDTO'] } = $props();
</script>

<section class="panel">
  <h2>历史风险估计</h2>
  <div class="stats-row">
    <MetricCard label="年化波动" metric={risk.annualized_volatility} /><MetricCard
      label="历史 VaR / 单日损失率"
      metric={risk.tail_risk.var}
    /><MetricCard label="历史 CVaR / 尾部平均损失率" metric={risk.tail_risk.cvar} />
  </div>
  <p class="context-note">
    置信度 {percent(risk.tail_risk.confidence)} · {risk.tail_risk.sample_count} 个收益观测 · {risk
      .tail_risk.quantile_method} · {risk.price_basis ?? '无价格口径'}。历史估计不代表未来损失上限。
  </p>
</section>
<section class="panel">
  <h2>资产与行业敞口</h2>
  <div class="stats-row">
    <div class="stat">
      <small>总权益 / 元</small><strong>{number(risk.exposure.total_value)}</strong>
    </div>
    <div class="stat">
      <small>现金权重</small><strong>{percent(risk.exposure.cash_weight)}</strong>
    </div>
    <div class="stat">
      <small>应收权重</small><strong>{percent(risk.exposure.receivables_weight)}</strong>
    </div>
    <div class="stat">
      <small>股票 HHI</small><strong>{number(risk.exposure.stock_hhi, 4)}</strong>
    </div>
  </div>
  <div class="two-columns">
    <div>
      <h3>证券权重</h3>
      <div class="table-scroll">
        <table>
          <thead><tr><th>证券</th><th>占总权益</th></tr></thead><tbody
            >{#each Object.entries(risk.exposure.asset_weights) as [id, weight] (id)}<tr
                ><td>{id}</td><td>{percent(weight)}</td></tr
              >{/each}</tbody
          >
        </table>
      </div>
    </div>
    <div>
      <h3>行业权重</h3>
      <div class="table-scroll">
        <table>
          <thead><tr><th>行业</th><th>占总权益</th></tr></thead><tbody
            >{#each Object.entries(risk.exposure.industry_weights) as [industry, weight] (industry)}<tr
                ><td>{industry}</td><td>{percent(weight)}</td></tr
              >{/each}</tbody
          >
        </table>
      </div>
    </div>
  </div>
  <p class="context-note">{risk.source}；分母包含现金与应收款，未识别行业归入“未分类”。</p>
  {#each [...risk.warnings, ...risk.exposure.warnings, ...risk.tail_risk.warnings] as note, index (index)}<p
      class="warning"
    >
      {note}
    </p>{/each}
</section>
