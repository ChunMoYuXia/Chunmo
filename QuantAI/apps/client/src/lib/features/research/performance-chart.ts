import type { components } from '$lib/api/schema';

export function performanceChartOptions(
  result: Pick<components['schemas']['BacktestResult'], 'nav' | 'benchmark'>,
) {
  const benchmark = new Map(result.benchmark?.map((point) => [point.session, point.nav]) ?? []);
  const axis = (name: string) => ({
    type: 'value' as const,
    scale: true,
    name,
    axisLabel: { color: '#a2b3ca' },
    splitLine: { lineStyle: { color: '#293b54' } },
  });
  return {
    animation: false,
    aria: { enabled: true },
    legend: { textStyle: { color: '#a2b3ca' } },
    tooltip: { trigger: 'axis' as const, renderMode: 'richText' as const },
    grid: { left: 80, right: result.benchmark ? 80 : 30, top: 50, bottom: 70 },
    xAxis: {
      type: 'category' as const,
      data: result.nav.map((point) => point.session),
      axisLabel: { color: '#a2b3ca' },
    },
    yAxis: [
      axis('组合权益 / 元'),
      { ...axis('基准原价 / 元'), show: Boolean(result.benchmark), splitLine: { show: false } },
    ],
    dataZoom: [{ type: 'inside' as const }, { type: 'slider' as const, bottom: 5, height: 20 }],
    series: [
      {
        name: '组合权益',
        type: 'line' as const,
        yAxisIndex: 0,
        showSymbol: false,
        sampling: 'lttb' as const,
        data: result.nav.map((point) => Number(point.nav)),
        lineStyle: { color: '#7eaeff', width: 2 },
      },
      ...(result.benchmark
        ? [
            {
              name: '基准原价（右轴）',
              type: 'line' as const,
              yAxisIndex: 1,
              showSymbol: false,
              sampling: 'lttb' as const,
              connectNulls: false,
              data: result.nav.map((point) =>
                benchmark.has(point.session) ? Number(benchmark.get(point.session)) : null,
              ),
              lineStyle: { color: '#d7b675', width: 1.5 },
            },
          ]
        : []),
    ],
  };
}
