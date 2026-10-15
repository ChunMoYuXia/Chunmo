<script lang="ts">
  import type { Preview } from '$lib/api/types';
  import { number } from '$lib/utils/format';
  let {
    preview,
    onCommit,
    busy = false,
  }: { preview: Preview; onCommit: () => void | Promise<void>; busy?: boolean } = $props();
</script>

<section class="import-preview" aria-label="导入预检结果">
  <div class="section-heading">
    <h3>预检结果</h3>
    <span class:bad={preview.errors.length > 0} class="badge"
      >{preview.errors.length
        ? `${preview.errors.length} 个问题`
        : preview.can_commit
          ? '可以提交'
          : '缺少必要数据'}</span
    >
  </div>
  <p class="muted">
    有效记录 {preview.row_count} 条，交易日历 {preview.calendar_sessions} 日。仅展示前 50 条。
  </p>
  {#each preview.errors as issue, index (index)}<p role="alert" class="issue">
      第 {issue.row} 行：{issue.message}
    </p>{/each}
  {#each preview.warnings as warning (warning)}<p class="warning">{warning}</p>{/each}
  {#if preview.rows.length}
    <div class="table-scroll">
      <table>
        <thead
          ><tr
            ><th>代码</th><th>名称</th><th>交易日</th><th class="numeric">开盘</th><th
              class="numeric">最高</th
            ><th class="numeric">最低</th><th class="numeric">收盘</th></tr
          ></thead
        ><tbody>
          {#each preview.rows as row (`${row.security_id}:${row.trade_date}`)}<tr
              ><td>{row.code}</td><td>{row.name}</td><td>{row.trade_date}</td><td class="numeric"
                >{number(row.open)}</td
              ><td class="numeric">{number(row.high)}</td><td class="numeric">{number(row.low)}</td
              ><td class="numeric">{number(row.close)}</td></tr
            >{/each}
        </tbody>
      </table>
    </div>
  {/if}
  <button
    class="primary"
    disabled={busy || preview.errors.length > 0 || !preview.can_commit}
    onclick={() => onCommit()}>提交导入</button
  >
</section>
