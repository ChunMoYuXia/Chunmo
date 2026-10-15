<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import RiskPanel from '$lib/features/research/RiskPanel.svelte';
  let portfolios = $state<components['schemas']['PortfolioDTO'][]>([]),
    id = $state(''),
    mode = $state<'holdings' | 'target'>('target'),
    confidence = $state(0.95),
    lookback = $state(60),
    loading = $state(false),
    failure = $state<unknown>(null),
    risk = $state<components['schemas']['PortfolioRiskDTO'] | null>(null);
  const scope = new RequestScope();
  const loadingScope = new RequestScope();
  onMount(() => {
    const ticket = loadingScope.start();
    request<components['schemas']['PortfolioDTO'][]>('/v1/portfolios', { signal: ticket.signal })
      .then((values) => {
        if (ticket.current()) {
          portfolios = values;
          id = page.url.searchParams.get('portfolio') ?? values[0]?.id ?? '';
        }
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      });
    return () => {
      loadingScope.cancel();
      scope.cancel();
    };
  });
  $effect(() => {
    void id;
    void mode;
    void confidence;
    void lookback;
    scope.cancel();
    loading = false;
    risk = null;
    failure = null;
  });
  async function analyze() {
    const ticket = scope.start();
    loading = true;
    failure = null;
    try {
      const result = await request<components['schemas']['PortfolioRiskDTO']>(
        '/v1/risk/portfolio',
        {
          method: 'POST',
          signal: ticket.signal,
          body: JSON.stringify({ portfolio_id: id, mode, confidence, lookback }),
        },
      );
      if (ticket.current()) risk = result;
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) loading = false;
    }
  }
</script>

<svelte:head><title>风险 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>组合风险</h1>
    <p>分别评估当前持仓和目标配置，保留数据缺口与方法说明。</p>
  </div>
</div>
<section class="panel">
  <div class="form-grid">
    <label
      >分析组合<select bind:value={id}
        ><option value="">选择已保存组合</option>{#each portfolios as item (item.id)}<option
            value={item.id}>{item.spec.name} · {item.spec.as_of.slice(0, 10)}</option
          >{/each}</select
      ></label
    ><label
      >分析口径<select bind:value={mode}
        ><option value="target">目标权重</option><option value="holdings">实际研究持仓</option
        ></select
      ></label
    ><label
      >置信度<select bind:value={confidence}
        ><option value={0.9}>90%</option><option value={0.95}>95%</option><option value={0.99}
          >99%</option
        ></select
      ></label
    ><label>历史窗口 / 交易日<input type="number" min="20" max="756" bind:value={lookback} /></label
    >
  </div>
  <button class="primary" disabled={!id || loading} onclick={() => void analyze()}
    >分析组合风险</button
  >
</section>
{#if loading}<StatusPanel title="正在计算历史风险" loading />{/if}{#if failure}<StatusPanel
    title="风险分析失败"
    error={failure}
  />{/if}{#if risk}<RiskPanel {risk} />{/if}
