<script lang="ts">
  import { onMount, type Snippet } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import {
    LayoutDashboard,
    Database,
    ChartCandlestick,
    ListFilter,
    Settings,
    BriefcaseBusiness,
    FlaskConical,
    ShieldCheck,
    Wallet,
    Files,
  } from 'lucide-svelte';
  import {
    chooseDataset,
    datasetCatalog,
    refreshDatasets,
    researchContext,
  } from '$lib/state/research';
  import { initializePreferences } from '$lib/state/preferences';
  import DatasetPicker from '$lib/features/data/DatasetPicker.svelte';
  import SecuritySearch from '$lib/features/market/SecuritySearch.svelte';
  import StatusPanel from './StatusPanel.svelte';
  let { children }: { children: Snippet } = $props();
  let failure = $state<unknown>(null);
  let busy = $state(false);
  let offline = $state(false);
  const navigation = [
    { path: '/', title: '总览', icon: LayoutDashboard },
    { path: '/data', title: '数据中心', icon: Database },
    { path: '/market', title: '行情', icon: ChartCandlestick },
    { path: '/screening', title: '选股', icon: ListFilter },
    { path: '/portfolios', title: '组合', icon: BriefcaseBusiness },
    { path: '/backtests', title: '回测', icon: FlaskConical },
    { path: '/risk', title: '风险', icon: ShieldCheck },
    { path: '/paper', title: '模拟盘', icon: Wallet },
    { path: '/reports', title: '报告', icon: Files },
    { path: '/settings', title: '设置', icon: Settings },
  ] as const;
  let dataset = $derived(
    $datasetCatalog.find((item) => item.dataset_id === $researchContext.datasetId),
  );
  async function reload() {
    busy = true;
    failure = null;
    try {
      await refreshDatasets();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
  onMount(() => {
    initializePreferences();
    void reload();
    const online = () => (offline = !navigator.onLine);
    online();
    window.addEventListener('online', online);
    window.addEventListener('offline', online);
    return () => {
      window.removeEventListener('online', online);
      window.removeEventListener('offline', online);
    };
  });
</script>

<a class="skip-link" href="#workspace">跳至研究内容</a>
<div class="app-shell">
  <aside class="sidebar">
    <a href={resolve('/')} class="brand"
      ><span class="brand-symbol" aria-hidden="true">▥</span><span
        ><strong>组合研究台</strong><small>本地研究 · 模拟验证</small></span
      ></a
    >
    <nav aria-label="主导航">
      {#each navigation as item (item.path)}<a
          href={resolve(item.path)}
          aria-current={page.url.pathname === item.path ? 'page' : undefined}
          ><item.icon size={18} strokeWidth={1.7} aria-hidden="true" /><span>{item.title}</span></a
        >{/each}
    </nav>
    <div class="sidebar-footer">
      <span class="local-mark"></span>本地工作空间<small>数据与持仓存于本机</small>
    </div>
  </aside>
  <div class="workspace-column">
    <header class="topbar">
      <div class="environment-controls">
        <DatasetPicker
          datasets={$datasetCatalog}
          selectedId={$researchContext.datasetId}
          onSelect={chooseDataset}
        />
        <label class="asof-picker"
          >截至<input
            type="date"
            aria-label="研究截至日期"
            value={$researchContext.asOf.slice(0, 10)}
            min={dataset?.first_session}
            max={dataset?.last_session}
            disabled={!dataset}
            onchange={(event) => {
              if (dataset && event.currentTarget.checkValidity())
                chooseDataset(dataset, event.currentTarget.value);
            }}
          /></label
        >
        {#if dataset}<span class:synthetic={dataset.source_kind === 'synthetic'} class="badge"
            >{dataset.source_kind === 'synthetic' ? '合成演示' : '本地导入'}</span
          >{/if}
      </div>
      <div class="header-search">
        <SecuritySearch
          id="global-security-search"
          context={$researchContext}
          onSelect={(security) =>
            void goto(resolve(`/market?security=${encodeURIComponent(security.security_id)}`))}
        />
      </div>
    </header>
    {#if dataset?.source_kind === 'synthetic'}<div class="source-strip">
        合成演示数据 <span>用于验证研究流程，不代表真实市场行情。</span>
      </div>{/if}
    {#if offline}<div class="source-strip">
        网络离线 <span>本地研究服务仍可使用；外部 AI 接入需要网络。</span>
      </div>{/if}
    <main id="workspace" class="workspace" tabindex="-1">
      {#if failure}<StatusPanel title="数据环境加载失败" error={failure} /><button
          disabled={busy}
          onclick={reload}>重新加载数据环境</button
        >{/if}
      {@render children()}
    </main>
  </div>
</div>
