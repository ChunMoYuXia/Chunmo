// @vitest-environment jsdom
import { expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import MetricCard from './MetricCard.svelte';

it('displays annualized ratio metrics in percent without changing server values', () => {
  render(MetricCard, {
    label: '年化波动',
    metric: {
      value: 0.12,
      unit: 'annualized_ratio',
      formula_version: 'metrics-v1',
      source: 'nav',
      as_of: null,
    },
  });
  expect(screen.getByText('12.00%')).toBeVisible();
});

it('keeps missing risk distinct from a zero-risk estimate', () => {
  render(MetricCard, {
    label: '历史 VaR',
    metric: {
      value: null,
      unit: 'ratio',
      formula_version: 'metrics-v1',
      source: 'historical',
      as_of: null,
      reason: 'all_cash_no_market_returns',
    },
  });
  expect(screen.getByText('—')).toBeVisible();
  expect(screen.getByText('all_cash_no_market_returns')).toBeVisible();
  expect(screen.queryByText('0.00%')).toBeNull();
});
