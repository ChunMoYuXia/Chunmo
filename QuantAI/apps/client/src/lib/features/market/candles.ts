import type { DailyBar } from '$lib/api/types';

export type Candle = {
  date: string;
  open: number;
  close: number;
  low: number;
  high: number;
  volume: number;
};

export function sampleCandles(bars: DailyBar[], maximum: number): Candle[] {
  const size = Math.max(1, Math.ceil(bars.length / Math.max(1, maximum)));
  const candles: Candle[] = [];
  for (let start = 0; start < bars.length; start += size) {
    const bucket = bars.slice(start, start + size);
    candles.push({
      date: bucket[bucket.length - 1].trade_date,
      open: Number(bucket[0].open),
      close: Number(bucket[bucket.length - 1].close),
      low: Math.min(...bucket.map((bar) => Number(bar.low))),
      high: Math.max(...bucket.map((bar) => Number(bar.high))),
      volume: bucket.reduce((sum, bar) => sum + bar.volume, 0),
    });
  }
  return candles;
}
