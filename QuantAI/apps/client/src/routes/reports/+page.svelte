<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import type { components } from '$lib/api/schema';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import { percent } from '$lib/utils/format';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import ReportView from '$lib/features/reports/ReportView.svelte';
  import Copilot from '$lib/features/ai/Copilot.svelte';
  let summaries = $state<components['schemas']['BacktestSummary'][]>([]),
    id = $state(''),
    report = $state<components['schemas']['ResearchReport'] | null>(null),
    failure = $state<unknown>(null),
    loading = $state(false);
  const scope = new RequestScope(),
    listScope = new RequestScope();
  onMount(() => {
    const ticket = listScope.start();
    request<components['schemas']['BacktestSummary'][]>('/v1/backtests', { signal: ticket.signal })
      .then((values) => {
        if (ticket.current()) {
          summaries = values;
          id = page.url.searchParams.get('research') ?? values[0]?.id ?? '';
        }
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      });
    return () => listScope.cancel();
  });
  $effect(() => {
    const selected = id;
    const ticket = scope.start();
    report = null;
    failure = null;
    loading = Boolean(selected);
    if (selected)
      request<components['schemas']['ResearchReport']>(
        `/v1/reports/${encodeURIComponent(selected)}`,
        { signal: ticket.signal },
      )
        .then((value) => {
          if (ticket.current()) report = value;
        })
        .catch((error) => {
          if (ticket.current()) failure = error;
        })
        .finally(() => {
          if (ticket.current()) loading = false;
        });
    return () => scope.cancel();
  });
</script>

<svelte:head><title>报告 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>证据报告</h1>
    <p>报告直接引用已保存研究，成本、指标与数据缺口均可追溯。</p>
  </div>
  <label
    >选择研究<select bind:value={id}
      ><option value="">选择已完成研究</option>{#each summaries as item (item.id)}<option
          value={item.id}
          >{item.created_at.slice(0, 10)} · {item.source_kind === 'synthetic'
            ? '合成演示'
            : '本地导入'} · {percent(item.total_return.value)}</option
        >{/each}</select
    ></label
  >
</div>
{#if failure}<StatusPanel title="报告加载失败" error={failure} />{/if}{#if loading}<StatusPanel
    title="正在构建证据报告"
    loading
  />{/if}
{#if report}<div class="workbench">
    <ReportView {report} reportId={id} />
    <aside class="stack">
      <Copilot reportId={id} /><StatusPanel
        title="报告输出范围"
        message="支持 Markdown、指标 CSV、原始权益序列与成交 CSV。PDF、Word 和全文检索将在后续阶段开发。"
      />
    </aside>
  </div>{:else if !loading}<StatusPanel
    title="还没有可展示的研究报告"
    message="先完成回测，再选择已保存的研究结果。"
  />{/if}
