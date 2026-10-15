import { expect, it } from 'vitest';
import { sampleCandles } from './candles';
import type { DailyBar } from '$lib/api/types';

it('aggregates visual candles without losing extremes or modifying original daily bars', () => {
  function bar(day: string, open: string, high: string, low: string, close: string): DailyBar {
    return {
      security_id: 's',
      trade_date: day,
      open,
      high,
      low,
      close,
      volume: 100,
      amount: '1000',
      available_at: `${day}T07:00:00Z`,
      suspended: false,
      limit_up: null,
      limit_down: null,
      adj_factor: null,
    };
  }
  const bars = [
    bar('2024-01-02', '10', '14', '8', '11'),
    bar('2024-01-03', '11', '13', '9', '12'),
    bar('2024-01-04', '12', '15', '9', '11'),
  ];
  const candles = sampleCandles(bars, 2);
  expect(candles).toEqual([
    { date: '2024-01-03', open: 10, close: 12, low: 8, high: 14, volume: 200 },
    { date: '2024-01-04', open: 12, close: 11, low: 9, high: 15, volume: 100 },
  ]);
  expect(bars[0].close).toBe('11');
});
