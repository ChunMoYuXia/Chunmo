<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import { datasetCatalog, researchContext } from '$lib/state/research';
  import { number, percent } from '$lib/utils/format';
  import StatusPanel from './StatusPanel.svelte';
  let portfolios = $state<components['schemas']['PortfolioDTO'][]>([]),
    results = $state<components['schemas']['BacktestSummary'][]>([]),
    accounts = $state<components['schemas']['AccountSnapshot'][]>([]),
    loading = $state(true),
    failure = $state<unknown>(null);
  let dataset = $derived(
    $datasetCatalog.find((value) => value.dataset_id === $researchContext.datasetId),
  );
  const scope = new RequestScope();
  async function reload() {
    const ticket = scope.start();
    loading = true;
    failure = null;
    try {
      const [p, r, a] = await Promise.all([
        request<components['schemas']['PortfolioDTO'][]>('/v1/portfolios', {
          signal: ticket.signal,
        }),
        request<components['schemas']['BacktestSummary'][]>('/v1/backtests', {
          signal: ticket.signal,
        }),
        request<components['schemas']['AccountSnapshot'][]>('/v1/paper/accounts', {
          signal: ticket.signal,
        }),
      ]);
      if (ticket.current()) {
        portfolios = p;
        results = r;
        accounts = a;
      }
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) loading = false;
    }
  }
  onMount(() => {
    void reload();
    return () => scope.cancel();
  });
</script>

<svelte:head><title>总览 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>研究总览</h1>
    <p>追踪本机保存的组合、研究和模拟账户。</p>
  </div>
  <button onclick={() => void reload()}>刷新总览</button>
</div>
{#if loading}<StatusPanel title="正在读取本地工作空间" loading />{/if}{#if failure}<StatusPanel
    title="工作空间读取失败"
    error={failure}
  />{/if}
{#if !dataset}<section class="panel">
    <h2>开始第一次本地研究</h2>
    <p class="muted">导入带交易日历的行情 CSV，或生成可复现的合成样本，然后筛选股票并建立组合。</p>
    <a href={resolve('/data')}>准备研究数据 →</a>
  </section>{:else}<section class="panel">
    <div class="section-heading">
      <h2>{dataset.name}</h2>
      <span class="badge"
        >{dataset.source_kind === 'synthetic' ? '合成演示' : '本地导入'} · v{dataset.version}</span
      >
    </div>
    <div class="stats-row">
      <div class="stat">
        <small>证券数量</small><strong>{dataset.security_count}</strong>
      </div>
      <div class="stat">
        <small>原始日线数量</small><strong>{number(dataset.row_count, 0)}</strong>
      </div>
      <div class="stat">
        <small>本机已保存组合</small><strong>{portfolios.length}</strong>
      </div>
      <div class="stat"><small>已完成研究</small><strong>{results.length}</strong></div>
    </div>
    <div class="actions">
      <a href={resolve('/market')}>历史行情 →</a><a href={resolve('/screening')}>筛选股票 →</a><a
        href={resolve('/portfolios')}>配置组合 →</a
      >
    </div>
    <p class="context-note">
      数据范围 {dataset.first_session} 至 {dataset.last_session}；组合与研究数量覆盖本机全部数据版本。
    </p>
  </section>{/if}
<div class="two-columns">
  <section class="panel">
    <div class="section-heading">
      <h2>近期研究</h2>
      <a href={resolve('/backtests')}>全部研究 →</a>
    </div>
    {#if !results.length}<StatusPanel
        title="还没有已完成的回测"
        message="先保存组合，再提交回测任务。"
      />{:else}<div class="table-scroll">
        <table>
          <thead><tr><th>研究时间</th><th>来源</th><th>收益率</th><th>最大回撤</th></tr></thead
          ><tbody
            >{#each results.slice(0, 5) as result (result.id)}<tr
                ><td>{result.created_at.slice(0, 10)}</td><td
                  >{result.source_kind === 'synthetic' ? '合成演示' : '本地导入'}</td
                ><td>{percent(result.total_return.value)}</td><td
                  >{percent(result.max_drawdown.value)}</td
                ></tr
              >{/each}</tbody
          >
        </table>
      </div>{/if}
  </section>
  <section class="panel">
    <div class="section-heading">
      <h2>模拟账户</h2>
      <a href={resolve('/paper')}>进入模拟盘 →</a>
    </div>
    {#if !accounts.length}<StatusPanel
        title="还没有模拟账户"
        message="创建账户，手动提交订单与推进交易日。"
      />{:else}{#each accounts.slice(0, 5) as account (account.id)}<div class="record-row">
          <div>
            <strong>{account.name}</strong><small
              >{account.session} · {account.source_kind === 'synthetic' ? '合成演示' : '本地导入'} · {account.status ===
              'paused'
                ? '暂停'
                : '活跃'}</small
            >
          </div>
          <span>{number(account.nav)} 元</span>
        </div>{/each}{/if}
  </section>
</div>
