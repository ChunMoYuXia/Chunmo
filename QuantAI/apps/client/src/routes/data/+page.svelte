<script lang="ts">
  import { request } from '$lib/api/client';
  import type { Dataset } from '$lib/api/types';
  import {
    chooseDataset,
    datasetCatalog,
    refreshDatasets,
    researchContext,
  } from '$lib/state/research';
  import { number } from '$lib/utils/format';
  import DataImport from '$lib/features/data/DataImport.svelte';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let seed = $state(42);
  let securities = $state(12);
  let sessions = $state(756);
  let busy = $state(false);
  let failure = $state<unknown>(null);
  let message = $state('');
  let selected = $derived(
    $datasetCatalog.find((dataset) => dataset.dataset_id === $researchContext.datasetId),
  );
  async function imported(dataset: Dataset) {
    await refreshDatasets();
    chooseDataset(dataset);
  }
  async function synthetic() {
    busy = true;
    failure = null;
    message = '';
    try {
      const dataset = await request<Dataset>('/v1/datasets/synthetic', {
        method: 'POST',
        body: JSON.stringify({ seed, securities, sessions }),
      });
      await imported(dataset);
      message = `演示数据已创建：${dataset.security_count} 只证券，${dataset.row_count} 条日线。`;
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<svelte:head><title>数据中心 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>数据中心</h1>
    <p>先检查原始数据，再固定版本开展研究。</p>
  </div>
  <span class="badge">原文与数据版本保存在本机</span>
</div>
<div class="two-columns">
  <DataImport onImported={imported} />
  <section class="panel">
    <div class="section-heading">
      <h2>创建合成演示数据</h2>
      <span class="badge synthetic">仅用于流程验证</span>
    </div>
    <p class="muted">固定随机种子可复现，包含停牌、涨跌停和现金分红场景。演示日历只排除周末。</p>
    <form
      onsubmit={(event) => {
        event.preventDefault();
        void synthetic();
      }}
    >
      <div class="form-grid">
        <label
          >证券数量<input type="number" min="1" max="200" bind:value={securities} required /></label
        ><label
          >交易日数量<input
            type="number"
            min="1"
            max="2520"
            bind:value={sessions}
            required
          /></label
        ><label
          >随机种子<input
            type="number"
            min="0"
            max="4294967295"
            bind:value={seed}
            required
          /></label
        >
      </div>
      <button class="primary" disabled={busy}>{busy ? '正在生成…' : '创建演示数据'}</button>
    </form>
    {#if failure}<StatusPanel
        title="创建演示数据失败"
        error={failure}
      />{/if}{#if message}<StatusPanel title="数据已就绪" {message} />{/if}
  </section>
</div>
<section class="panel">
  <div class="section-heading">
    <h2>数据版本</h2>
    <small>共 {$datasetCatalog.length} 个固定版本</small>
  </div>
  {#if !$datasetCatalog.length}<StatusPanel
      title="还没有研究数据"
      message="导入 CSV 与交易日历，或用合成数据走通研究流程。"
    />{:else}<div class="table-scroll">
      <table>
        <thead
          ><tr
            ><th>数据集</th><th>来源</th><th>版本</th><th class="numeric">证券数</th><th
              class="numeric">日线条数</th
            ><th>区间</th><th>操作</th></tr
          ></thead
        ><tbody
          >{#each $datasetCatalog as dataset (dataset.dataset_id)}<tr
              ><td>{dataset.name}</td><td
                ><span class:synthetic={dataset.source_kind === 'synthetic'} class="badge"
                  >{dataset.source_kind === 'synthetic' ? '合成演示' : '本地 CSV'}</span
                ></td
              ><td>v{dataset.version}</td><td class="numeric"
                >{number(dataset.security_count, 0)}</td
              ><td class="numeric">{number(dataset.row_count, 0)}</td><td
                >{dataset.first_session} 至 {dataset.last_session}</td
              ><td
                ><button
                  disabled={selected?.dataset_id === dataset.dataset_id}
                  onclick={() => chooseDataset(dataset)}
                  >{selected?.dataset_id === dataset.dataset_id
                    ? '正在使用'
                    : '设为研究数据'}</button
                ></td
              ></tr
            >{/each}</tbody
        >
      </table>
    </div>{/if}
</section>
{#if selected}<section class="panel">
    <h2>来源与数据口径</h2>
    <dl class="provenance">
      <dt>数据集</dt>
      <dd>{selected.name} / v{selected.version}</dd>
      <dt>最后交易日</dt>
      <dd>{selected.last_session}</dd>
      <dt>入库时间</dt>
      <dd>{new Date(selected.ingested_at).toLocaleString('zh-CN')}</dd>
      <dt>日历版本</dt>
      <dd>{selected.calendar_id}</dd>
      <dt>数据摘要</dt>
      <dd>{selected.content_hash}</dd>
      <dt>解析 / 生成器</dt>
      <dd>
        {selected.parser_version ??
          selected.generator_version ??
          '未提供'}{#if selected.generator_seed !== null}，种子 {selected.generator_seed}{/if}
      </dd>
      {#each selected.source_files as source (source.role)}<dt>
          {source.role === 'quotes' ? '行情原文摘要' : '日历原文摘要'}
        </dt>
        <dd>{source.content_hash}</dd>{/each}
    </dl>
    {#each selected.warnings as warning (warning)}<p class="warning">{warning}</p>{/each}
  </section>{/if}
