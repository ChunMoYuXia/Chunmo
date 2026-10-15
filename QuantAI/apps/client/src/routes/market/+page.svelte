<script lang="ts">
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import { downloadText } from '$lib/utils/download';
  import type { DailyBar, Feature } from '$lib/api/types';
  import { marketQuery, researchContext } from '$lib/state/research';
  import { number, percent } from '$lib/utils/format';
  import SecuritySearch from '$lib/features/market/SecuritySearch.svelte';
  import FeatureTable from '$lib/features/market/FeatureTable.svelte';
  import PriceChart from '$lib/features/market/PriceChart.svelte';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let features = $state<Feature[]>([]);
  let selectedId = $state('');
  let bars = $state<DailyBar[]>([]);
  let busy = $state(false);
  let exporting = $state(false);
  let failure = $state<unknown>(null);
  const listScope = new RequestScope();
  const quoteScope = new RequestScope();
  let selected = $derived(features.find((item) => item.security_id === selectedId));
  let latest = $derived(bars[bars.length - 1]);
  $effect(() => {
    const requested = page.url.searchParams.get('security');
    if (requested) selectedId = requested;
  });
  $effect(() => {
    const context = $researchContext;
    const ticket = listScope.start();
    features = [];
    if (!context.datasetId) {
      selectedId = '';
      return () => listScope.cancel();
    }
    request<Feature[]>(`/v1/market/snapshot?${marketQuery(context)}`, { signal: ticket.signal })
      .then((rows) => {
        if (!ticket.current()) return;
        features = rows;
        selectedId =
          page.url.searchParams.get('security') ??
          (rows.some((row) => row.security_id === selectedId)
            ? selectedId
            : (rows[0]?.security_id ?? ''));
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      });
    return () => listScope.cancel();
  });
  $effect(() => {
    const context = $researchContext;
    const securityId = selectedId;
    const ticket = quoteScope.start();
    bars = [];
    failure = null;
    busy = Boolean(context.datasetId && securityId);
    if (!context.datasetId || !securityId) return () => quoteScope.cancel();
    request<DailyBar[]>(
      `/v1/market/kline?${marketQuery(context, { security_id: securityId, limit: '5000' })}`,
      { signal: ticket.signal },
    )
      .then((rows) => {
        if (ticket.current()) bars = rows;
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      })
      .finally(() => {
        if (ticket.current()) busy = false;
      });
    return () => quoteScope.cancel();
  });
  async function exportQuotes() {
    exporting = true;
    failure = null;
    try {
      const file = await request<components['schemas']['ExportDocument']>(
        `/v1/market/export?${marketQuery($researchContext, { security_id: selectedId })}`,
      );
      downloadText(file.filename, file.content, file.media_type);
    } catch (error) {
      failure = error;
    } finally {
      exporting = false;
    }
  }
</script>

<svelte:head><title>行情 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>历史行情</h1>
    <p>查看原始价格、交易量与完整窗口的历史特征。</p>
  </div>
  <div style="width:340px;max-width:100%">
    <SecuritySearch
      id="market-security-search"
      context={$researchContext}
      onSelect={(security) =>
        void goto(resolve(`/market?security=${encodeURIComponent(security.security_id)}`))}
    />
  </div>
</div>
{#if !$researchContext.datasetId}<StatusPanel
    title="先选择研究数据"
    message="在数据中心导入行情，或创建演示数据。"
  />{:else}
  {#if selected}<section class="stats-row">
      <div class="stat">
        <small>{selected.code} / {selected.industry ?? '未分类'}</small><strong
          style="font-size:19px">{selected.name}</strong
        >
      </div>
      <div class="stat"><small>原始收盘价 / 元</small><strong>{number(latest?.close)}</strong></div>
      <div class="stat">
        <small>20日动量</small><strong
          class:gain={(selected.momentum_20d ?? 0) > 0}
          class:loss={(selected.momentum_20d ?? 0) < 0}>{percent(selected.momentum_20d)}</strong
        >
      </div>
      <div class="stat">
        <small>年化波动 / 244 日</small><strong>{percent(selected.volatility_20d)}</strong>
      </div>
    </section>{/if}
  <section class="panel">
    <div class="actions">
      <button disabled={exporting || !bars.length} onclick={() => void exportQuotes()}
        >导出原始日线 CSV</button
      >
    </div>
    {#if busy}<StatusPanel title="正在加载历史日线" loading />{:else if !bars.length}<StatusPanel
        title="此时点没有可见行情"
        message="请检查证券、截至日期或原始数据缺口。"
      />{/if}<PriceChart {bars} name={selected?.name ?? selectedId} />
    <p class="context-note">
      公开历史模式，截至 {$researchContext.asOf.slice(0, 10)}；最后可见交易日 {latest?.trade_date ??
        '未知'}。复权及财务修订历史未提供时，不作额外假设。
    </p>
    {#each selected?.warnings ?? [] as warning (warning)}<p class="warning">{warning}</p>{/each}
  </section>
  <section class="panel">
    <div class="section-heading">
      <h2>市场快照</h2>
      <small>{features.length} 只证券</small>
    </div>
    <FeatureTable {features} />
  </section>
{/if}
{#if failure}<StatusPanel title="行情加载失败" error={failure} />{/if}
