<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import { datasetCatalog } from '$lib/state/research';
  import { percent, number } from '$lib/utils/format';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import BacktestForm from './BacktestForm.svelte';
  import BacktestResultView from './BacktestResultView.svelte';
  import JobProgress from './JobProgress.svelte';
  type Portfolio = components['schemas']['PortfolioDTO'];
  let portfolios = $state<Portfolio[]>([]),
    summaries = $state<components['schemas']['BacktestSummary'][]>([]),
    jobs = $state<components['schemas']['JobRecord'][]>([]);
  let portfolioId = $state(''),
    jobId = $state(''),
    artifact = $state<components['schemas']['ResearchArtifact'] | null>(null),
    failure = $state<unknown>(null),
    loading = $state(true);
  const scope = new RequestScope();
  let portfolio = $derived(portfolios.find((p) => p.id === portfolioId));
  let dataset = $derived($datasetCatalog.find((d) => d.dataset_id === portfolio?.spec.dataset_id));
  async function reload() {
    const ticket = scope.start();
    loading = true;
    failure = null;
    try {
      const [p, s, j] = await Promise.all([
        request<Portfolio[]>('/v1/portfolios', { signal: ticket.signal }),
        request<components['schemas']['BacktestSummary'][]>('/v1/backtests', {
          signal: ticket.signal,
        }),
        request<components['schemas']['JobRecord'][]>('/v1/jobs', { signal: ticket.signal }),
      ]);
      if (ticket.current()) {
        portfolios = p;
        summaries = s;
        jobs = j;
        if (!portfolioId) portfolioId = page.url.searchParams.get('portfolio') ?? p[0]?.id ?? '';
      }
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) loading = false;
    }
  }
  async function view(id: string) {
    const ticket = scope.start();
    failure = null;
    try {
      const result = await request<components['schemas']['ResearchArtifact']>(
        `/v1/backtests/${encodeURIComponent(id)}`,
        { signal: ticket.signal },
      );
      if (ticket.current()) {
        jobId = '';
        artifact = result;
      }
    } catch (error) {
      if (ticket.current()) failure = error;
    }
  }
  onMount(() => {
    void reload();
    return () => scope.cancel();
  });
</script>

<svelte:head><title>回测 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>回测研究</h1>
    <p>真实撮合约束、完整成本与可复现的计算版本。</p>
  </div>
  <button onclick={() => void reload()}>刷新记录</button>
</div>
{#if loading}<StatusPanel title="正在加载研究记录" loading />{/if}
{#if failure}<StatusPanel title="研究记录加载失败" error={failure} />{/if}
<div class="workbench">
  <div class="stack">
    {#if jobId}<JobProgress
        {jobId}
        onResult={(value) => {
          artifact = value;
          void reload();
        }}
      />{/if}{#if artifact}<BacktestResultView {artifact} />{:else if !jobId}<StatusPanel
        title="尚未选择研究结果"
        message="选择已保存组合提交回测，或查看历史研究。"
      />{/if}
    <section class="panel">
      <h2>历史研究</h2>
      <div class="table-scroll">
        <table>
          <thead
            ><tr
              ><th>时间</th><th>来源</th><th>区间收益</th><th>最大回撤</th><th>费用 / 元</th><th
                >操作</th
              ></tr
            ></thead
          ><tbody
            >{#each summaries as item (item.id)}<tr
                ><td>{item.created_at.slice(0, 19).replace('T', ' ')}</td><td
                  >{item.source_kind === 'synthetic' ? '合成演示' : '本地导入'} · v{item.data_version}</td
                ><td>{percent(item.total_return.value)}</td><td
                  >{percent(item.max_drawdown.value)}</td
                ><td>{number(item.costs)}</td><td
                  ><button onclick={() => void view(item.id)}>查看结果</button></td
                ></tr
              >{/each}</tbody
          >
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>任务记录</h2>
      {#each jobs as job (job.id)}<div class="record-row">
          <div>
            <code>{job.id.slice(0, 12)}</code><small
              >{job.state} · {job.progress.completed}/{job.progress.total}</small
            >
          </div>
          <button
            onclick={() => {
              artifact = null;
              jobId = job.id;
            }}>查询任务</button
          >
        </div>{/each}
    </section>
  </div>
  <aside class="stack">
    <section class="panel">
      <label
        >回测组合<select bind:value={portfolioId}
          ><option value="">选择组合</option>{#each portfolios as item (item.id)}<option
              value={item.id}>{item.spec.name}</option
            >{/each}</select
        ></label
      >
    </section>
    {#if portfolio && dataset}<BacktestForm
        {portfolio}
        {dataset}
        onSubmit={(id) => {
          artifact = null;
          jobId = id;
          void reload();
        }}
      />{:else}<StatusPanel
        title="需要已保存的组合和数据版本"
        message="先到组合页配置并保存组合。"
      />{/if}
  </aside>
</div>
