<script lang="ts">
  import type { components } from '$lib/api/schema';
  import { request } from '$lib/api/client';
  import { downloadText } from '$lib/utils/download';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let { report, reportId }: { report: components['schemas']['ResearchReport']; reportId: string } =
    $props();
  let busy = $state(false),
    failure = $state<unknown>(null);
  async function exportReport(format: 'markdown' | 'metrics' | 'nav' | 'trades') {
    busy = true;
    failure = null;
    try {
      const file = await request<components['schemas']['ExportDocument']>(
        `/v1/reports/${encodeURIComponent(reportId)}/export?format=${format}`,
      );
      downloadText(file.filename, file.content, file.media_type);
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel">
  <div class="section-heading">
    <div>
      <h2>规则分析报告</h2>
      <p class="context-note">
        {report.version} · {report.kind} · 模型：未使用 · 截至 {report.as_of.slice(0, 10)}
      </p>
    </div>
    <span class="badge">确定性指标与证据</span>
  </div>
  <div class="actions">
    <button disabled={busy} onclick={() => void exportReport('markdown')}>导出 Markdown</button
    ><button disabled={busy} onclick={() => void exportReport('metrics')}>导出指标 CSV</button
    ><button disabled={busy} onclick={() => void exportReport('nav')}>导出原始权益序列</button
    ><button disabled={busy} onclick={() => void exportReport('trades')}>导出成交 CSV</button>
  </div>
  {#if failure}<StatusPanel title="报告导出失败" error={failure} />{/if}
  {#each report.sections as section, index (index)}<article class="report-section">
      <h3>{section.title}</h3>
      <p style="white-space:pre-line">{section.text || '未提供附加说明。'}</p>
      <small
        >引用：{#each section.evidence_ids as citation (citation)}<a
            class="evidence-link"
            href={`#evidence-${citation}`}>{citation.slice(0, 12)}</a
          >{/each}</small
      >
    </article>{/each}
  <h3>证据索引</h3>
  {#each report.evidence as evidence (evidence.id)}<div
      id={`evidence-${evidence.id}`}
      class="evidence-record"
    >
      <strong>{evidence.source}</strong>
      <p class="hash">{evidence.data_version}</p>
      <p class="context-note">{evidence.as_of} · {evidence.formula_version}</p>
      {#each evidence.details ?? [] as detail (detail)}<p class="context-note">{detail}</p>{/each}
    </div>{/each}
</section>
