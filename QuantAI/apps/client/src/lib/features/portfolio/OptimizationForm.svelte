<script lang="ts">
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import type { ResearchContext } from '$lib/state/research';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let {
    context,
    universe,
    maxWeight,
    knowledgeMode = 'public_pit',
    onResult,
  }: {
    context: ResearchContext;
    universe: string[];
    maxWeight: string;
    knowledgeMode?: 'public_pit' | 'system_replay';
    onResult: (value: components['schemas']['OptimizationDTO']) => void;
  } = $props();
  let method = $state<'equal' | 'inverse_volatility' | 'minimum_variance'>('equal'),
    lookback = $state(60),
    busy = $state(false),
    failure = $state<unknown>(null);
  const scope = new RequestScope();
  $effect(() => {
    void context;
    void universe;
    void maxWeight;
    void knowledgeMode;
    scope.cancel();
    busy = false;
    failure = null;
  });
  async function optimize() {
    const ticket = scope.start();
    busy = true;
    failure = null;
    try {
      const result = await request<components['schemas']['OptimizationDTO']>(
        '/v1/portfolios/optimize',
        {
          method: 'POST',
          signal: ticket.signal,
          body: JSON.stringify({
            dataset_id: context.datasetId,
            as_of: context.asOf,
            universe,
            max_weight: maxWeight,
            method,
            lookback,
            knowledge_mode: knowledgeMode,
          }),
        },
      );
      if (ticket.current()) onResult(result);
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) busy = false;
    }
  }
</script>

<div class="form-grid">
  <label
    >配置方法<select bind:value={method}
      ><option value="equal">等权</option><option value="inverse_volatility">波动率倒数</option
      ><option value="minimum_variance">最小方差</option></select
    ></label
  ><label>历史窗口 / 交易日<input type="number" min="20" max="756" bind:value={lookback} /></label>
</div>
<button disabled={busy || !universe.length} onclick={() => void optimize()}
  >{busy ? '正在计算…' : '计算目标权重'}</button
>
<p class="context-note">波动率倒数按单资产波动分配；最小方差使用收缩协方差矩阵。</p>
{#if failure}<StatusPanel title="无法满足配置约束" error={failure} />{/if}
