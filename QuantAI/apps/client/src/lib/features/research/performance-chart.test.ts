import { expect, it } from 'vitest';
import { performanceChartOptions } from './performance-chart';
it('keeps benchmark raw prices on a separate labelled axis and leaves gaps unfilled', () => {
  const options = performanceChartOptions({
    nav: [
      { session: '2025-01-02', nav: '100000', cash: '0', receivables: '0', stale_securities: [] },
      { session: '2025-01-03', nav: '110000', cash: '0', receivables: '0', stale_securities: [] },
    ],
    benchmark: [
      { session: '2025-01-02', nav: '10', cash: '0', receivables: '0', stale_securities: [] },
    ],
  });
  expect(options.series[0].data).toEqual([100000, 110000]);
  expect(options.series[1].data).toEqual([10, null]);
  expect(options.series[1].yAxisIndex).toBe(1);
  expect(options.yAxis[1].name).toBe('基准原价 / 元');
});
