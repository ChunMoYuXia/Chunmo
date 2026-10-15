<script lang="ts">
  import { onMount } from 'svelte';
  import { performanceChartOptions } from './performance-chart';
  import type { ECharts } from 'echarts/core';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let { result }: { result: components['schemas']['BacktestResult'] } = $props();
  let container: HTMLDivElement;
  let chart = $state.raw<ECharts | null>(null);
  let failure = $state<unknown>(null);
  onMount(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    import('$lib/components/chart-runtime')
      .then(({ initChart }) => {
        if (disposed) return;
        chart = initChart(container);
        observer = new ResizeObserver(() => chart?.resize());
        observer.observe(container);
      })
      .catch((error) => (failure = error));
    return () => {
      disposed = true;
      observer?.disconnect();
      chart?.dispose();
    };
  });
  $effect(() => {
    if (!chart) return;
    chart.setOption(performanceChartOptions(result), true);
  });
</script>

<div
  bind:this={container}
  style="width:100%;height:340px"
  role="img"
  aria-label="组合权益与基准原价双轴曲线"
></div>
{#if failure}<StatusPanel title="图表加载失败" error={failure} />{/if}
<p class="context-note">
  按视窗采样绘图；原始日序列保存在研究结果中。基准显示原始价格（右轴），不含分红，缺失不补零。
</p>
