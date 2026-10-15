<script lang="ts">
  import type { ScreeningStep } from '$lib/api/types';
  let {
    steps,
    universeCount,
    matchingCount,
    onRollback,
  }: {
    steps: ScreeningStep[];
    universeCount: number;
    matchingCount: number;
    onRollback: (prefixLength: number) => void;
  } = $props();
  const reasons: Record<string, string> = {
    missing_value: '数据缺失',
    below_threshold: '低于门槛',
    above_threshold: '高于门槛',
    not_equal: '不符合条件',
    not_in: '不在指定范围',
  };
</script>

<section aria-label="筛选漏斗" class="funnel">
  <div class="section-heading">
    <h3>筛选归因</h3>
    <strong>{universeCount} → {matchingCount}</strong>
  </div>
  {#if !steps.length}<p class="muted">未添加条件，保留原始候选池。</p>{/if}
  {#each steps as step, index (step.index)}<div class="funnel-step">
      <div class="section-heading">
        <strong>条件 {index + 1}</strong><span>{step.input_count} → {step.remaining_count}</span>
      </div>
      <div class="funnel-track">
        <span style:width={`${universeCount ? (step.remaining_count / universeCount) * 100 : 0}%`}
        ></span>
      </div>
      <div class="funnel-reasons">
        {#each Object.entries(step.exclusions) as [reason, count] (reason)}<span
            >{reasons[reason] ?? reason}：{count}</span
          >{/each}
      </div>
      <button
        class="text-button"
        aria-label={`撤回条件 ${index + 1}`}
        onclick={() => onRollback(index)}>撤回此条件及后续条件</button
      >
    </div>{/each}
</section>
