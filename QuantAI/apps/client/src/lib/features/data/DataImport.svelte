<script lang="ts">
  import { request } from '$lib/api/client';
  import type { Dataset, Preview } from '$lib/api/types';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import ImportPreview from './ImportPreview.svelte';
  let { onImported }: { onImported: (dataset: Dataset) => void | Promise<void> } = $props();
  let quotes = $state<File | null>(null);
  let calendar = $state<File | null>(null);
  let unit = $state('shares');
  let mapping = $state<Record<string, string>>({});
  let preview = $state<Preview | null>(null);
  let busy = $state(false);
  let failure = $state<unknown>(null);
  let submitted = $state('');
  const fields = [
    ['code', '证券代码'],
    ['date', '交易日'],
    ['open', '开盘价'],
    ['high', '最高价'],
    ['low', '最低价'],
    ['close', '收盘价'],
    ['volume', '成交量'],
    ['name', '证券名称'],
    ['exchange', '交易所'],
    ['industry', '行业'],
    ['amount', '成交额'],
    ['limit_up', '涨停价'],
    ['limit_down', '跌停价'],
  ];

  async function preflight() {
    if (!quotes) return;
    busy = true;
    failure = null;
    submitted = '';
    preview = null;
    const body = new FormData();
    body.set('file', quotes);
    body.set('volume_unit', unit);
    if (calendar) body.set('calendar_file', calendar);
    body.set(
      'mapping',
      JSON.stringify(
        Object.fromEntries(
          Object.entries(mapping)
            .filter(([, column]) => column.trim())
            .map(([field, column]) => [column.trim(), field]),
        ),
      ),
    );
    try {
      preview = await request<Preview>('/v1/datasets/import/preview', { method: 'POST', body });
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }

  async function commit() {
    if (!preview?.can_commit || preview.errors.length || busy) return;
    busy = true;
    failure = null;
    try {
      const dataset = await request<Dataset>(`/v1/datasets/import/${preview.id}/commit`, {
        method: 'POST',
      });
      submitted = `已提交 ${dataset.row_count} 条记录，数据版本 v${dataset.version}`;
      preview = null;
      await onImported(dataset);
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel">
  <div class="section-heading">
    <h2>导入日线数据</h2>
    <span class="badge">本地 CSV</span>
  </div>
  <p class="muted">支持 UTF-8 / GB18030 和常用中文列名。真实行情需要同时提供交易日历。</p>
  <form
    onsubmit={(event) => {
      event.preventDefault();
      void preflight();
    }}
  >
    <div class="form-grid">
      <label
        >行情 CSV<input
          type="file"
          accept=".csv"
          disabled={busy}
          onchange={(event) => {
            quotes = event.currentTarget.files?.[0] ?? null;
            preview = null;
          }}
        /></label
      >
      <label
        >交易日历 CSV<input
          type="file"
          accept=".csv"
          disabled={busy}
          onchange={(event) => {
            calendar = event.currentTarget.files?.[0] ?? null;
            preview = null;
          }}
        /></label
      >
      <label
        >成交量单位<select bind:value={unit} disabled={busy} onchange={() => (preview = null)}
          ><option value="shares">股</option><option value="lots">手（每手 100 股）</option></select
        ></label
      >
    </div>
    <details>
      <summary>自定义字段映射</summary>
      <p class="muted">自动识别失败时，填写该字段在文件中的列名；留空使用自动识别。</p>
      <div class="form-grid">
        {#each fields as [key, label] (key)}<label
            >{label}<input
              value={mapping[key] ?? ''}
              disabled={busy}
              placeholder="文件列名"
              oninput={(event) => {
                mapping[key] = event.currentTarget.value;
                preview = null;
              }}
            /></label
          >{/each}
      </div>
    </details>
    <button class="primary" type="submit" disabled={!quotes || busy}
      >{busy ? '正在处理…' : '预检数据'}</button
    >
  </form>
  {#if failure}<StatusPanel error={failure} />{/if}
  {#if submitted}<StatusPanel title="导入完成" message={submitted} />{/if}
  {#if preview}<ImportPreview {preview} onCommit={commit} {busy} />{/if}
</section>
