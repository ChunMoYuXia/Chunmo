<script lang="ts">
  import type { components } from '$lib/api/schema';
  import { number, percent } from '$lib/utils/format';
  let { label, metric }: { label: string; metric: components['schemas']['Metric'] } = $props();
</script>

<div class="stat">
  <small>{label}</small><strong
    >{['ratio', 'annualized_ratio', 'one_day_loss_ratio'].includes(metric.unit)
      ? percent(metric.value)
      : number(metric.value)}</strong
  >
  {#if metric.value === null}<small class="warning">{metric.reason ?? '数据缺失'}</small>{/if}
  <small title={`${metric.source} · ${metric.as_of ?? '无时间'}`}
    >{metric.formula_version} · {metric.unit}</small
  >
</div>
