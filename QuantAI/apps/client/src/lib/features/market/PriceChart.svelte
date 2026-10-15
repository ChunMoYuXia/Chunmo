<script lang="ts">
  import { onMount } from 'svelte';
  import type { ECharts } from 'echarts/core';
  import type { DailyBar } from '$lib/api/types';
  import { preferences } from '$lib/state/preferences';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import { sampleCandles } from './candles';
  let { bars, name }: { bars: DailyBar[]; name: string } = $props();
  let container: HTMLDivElement;
  let chart = $state.raw<ECharts | null>(null);
  let width = $state(900);
  let period = $state(126);
  let failure = $state<unknown>(null);
  let visible = $derived(period ? bars.slice(-period) : bars);
  let candles = $derived(sampleCandles(visible, Math.max(40, Math.floor(width / 6))));
  onMount(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    const resize = () => {
      width = container.clientWidth || 900;
      chart?.resize();
    };
    import('$lib/components/chart-runtime')
      .then(({ initChart }) => {
        if (disposed) return;
        chart = initChart(container);
        resize();
        if (typeof ResizeObserver !== 'undefined') {
          observer = new ResizeObserver(resize);
          observer.observe(container);
        } else window.addEventListener('resize', resize);
      })
      .catch((error) => (failure = error));
    return () => {
      disposed = true;
      observer?.disconnect();
      window.removeEventListener('resize', resize);
      chart?.dispose();
    };
  });
  $effect(() => {
    const instance = chart;
    const rows = candles;
    const redUp = $preferences.upColor === 'red';
    if (!instance) return;
    if (!rows.length) {
      instance.clear();
      return;
    }
    const gain = redUp ? '#ff828b' : '#64d4ad';
    const loss = redUp ? '#64d4ad' : '#ff828b';
    instance.setOption(
      {
        animation: false,
        aria: { enabled: true, label: { description: `${name} 原始价格日线，人民币元，成交量股` } },
        tooltip: { trigger: 'axis', renderMode: 'richText', axisPointer: { type: 'cross' } },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        grid: [
          { left: 55, right: 24, top: 25, height: '54%' },
          { left: 55, right: 24, top: '73%', height: '14%' },
        ],
        xAxis: [
          {
            type: 'category',
            data: rows.map((row) => row.date),
            axisLabel: { show: false },
            axisLine: { lineStyle: { color: '#3b506e' } },
          },
          {
            type: 'category',
            gridIndex: 1,
            data: rows.map((row) => row.date),
            axisLabel: { color: '#a2b3ca', fontSize: 10 },
            axisLine: { lineStyle: { color: '#3b506e' } },
          },
        ],
        yAxis: [
          {
            scale: true,
            name: '元',
            nameTextStyle: { color: '#a2b3ca' },
            axisLabel: { color: '#a2b3ca', fontSize: 10 },
            splitLine: { lineStyle: { color: '#293b54' } },
          },
          {
            gridIndex: 1,
            name: '股',
            nameTextStyle: { color: '#a2b3ca' },
            axisLabel: {
              color: '#a2b3ca',
              fontSize: 10,
              formatter: (value: number) =>
                value >= 10000 ? `${Math.round(value / 10000)}万` : String(value),
            },
            splitLine: { show: false },
          },
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1] },
          {
            type: 'slider',
            xAxisIndex: [0, 1],
            bottom: 0,
            height: 18,
            borderColor: '#3b506e',
            textStyle: { color: '#a2b3ca' },
            fillerColor: '#366cb344',
          },
        ],
        series: [
          {
            name: '原始价格',
            type: 'candlestick',
            data: rows.map((row) => [row.open, row.close, row.low, row.high]),
            itemStyle: { color: gain, color0: loss, borderColor: gain, borderColor0: loss },
          },
          {
            name: '成交量',
            type: 'bar',
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: rows.map((row) => ({
              value: row.volume,
              itemStyle: { color: row.close >= row.open ? gain : loss, opacity: 0.65 },
            })),
          },
        ],
      },
      { notMerge: true, lazyUpdate: true },
    );
  });
</script>

<div class="section-heading">
  <h2>历史日线</h2>
  <label class="actions"
    >区间<select aria-label="图表区间" bind:value={period}
      ><option value={63}>近 3 个月</option><option value={126}>近 6 个月</option><option
        value={252}>近 1 年</option
      ><option value={0}>全部已载入数据</option></select
    ></label
  >
</div>
{#if failure}<StatusPanel title="图表加载失败" error={failure} />{/if}
<div
  class="chart-container"
  bind:this={container}
  role="img"
  aria-label={`${name} 原始日线价格与成交量`}
></div>
<p class="chart-caption">
  原始价格 / 人民币元，成交量 / 股。{#if candles.length < visible.length}当前视窗聚合 {visible.length}
    日为 {candles.length} 根柱，保留最高与最低点。{:else}当前显示 {visible.length} 根日线。{/if}原始日线仍完整保留，图表不用于撮合。
</p>
