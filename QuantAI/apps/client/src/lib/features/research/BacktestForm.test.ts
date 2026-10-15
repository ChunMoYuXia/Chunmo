// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import BacktestForm from './BacktestForm.svelte';
afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});
it('preserves edited dates when the same saved portfolio is refreshed', async () => {
  const props = {
    portfolio: {
      id: 'p1',
      created_at: '2025-01-31T07:00:00Z',
      cash_weight: '0',
      spec: {
        knowledge_mode: 'public_pit' as const,
        name: '组合',
        dataset_id: 'd1',
        as_of: '2025-01-31T07:00:00Z',
        target_weights: { 'XSHE:000001': '1' },
        max_weight: '1',
        cash: '100000',
      },
    },
    dataset: {
      dataset_id: 'd1',
      calendar_id: 'cal1',
      first_session: '2024-01-02',
      last_session: '2025-01-31',
    },
    onSubmit: () => {},
  };
  const view = render(BacktestForm, props);
  await fireEvent.input(screen.getByLabelText('结束日期'), { target: { value: '2024-01-03' } });
  await view.rerender({
    ...props,
    portfolio: { ...props.portfolio },
    dataset: { ...props.dataset },
  });
  expect(screen.getByLabelText('结束日期')).toHaveValue('2024-01-03');
  await view.rerender({ ...props, portfolio: { ...props.portfolio, id: 'p2' } });
  expect(screen.getByLabelText('结束日期')).toHaveValue('2025-01-31');
});

it('submits saved portfolio weights with explicit calendar, benchmark and cost assumptions', async () => {
  configureConnection({ url: '', token: '' });
  let received: unknown;
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    received = JSON.parse(init.body as string);
    return Response.json({ data: { task_id: 'j1', state: 'queued' } });
  });
  let job = '';
  render(BacktestForm, {
    portfolio: {
      id: 'p',
      spec: {
        knowledge_mode: 'public_pit',
        name: '组合',
        dataset_id: 'd1',
        as_of: '2025-01-31T07:00:00Z',
        target_weights: { '600001.SH': '0.8' },
        max_weight: '1',
        cash: '100000',
      },
      created_at: '2025-01-31T07:00:00Z',
      cash_weight: '0.2',
    },
    dataset: {
      dataset_id: 'd1',
      calendar_id: 'cal1',
      first_session: '2024-01-02',
      last_session: '2025-01-31',
    },
    onSubmit: (id) => (job = id),
  });
  await userEvent.click(screen.getByRole('button', { name: '提交回测' }));
  expect(received).toMatchObject({
    calendar_id: 'cal1',
    dataset_id: 'd1',
    weights: { '600001.SH': '0.8' },
    capital: '100000',
    slippage_bps: '8',
    require_price_limits: true,
    benchmark_id: null,
    annualization: 244,
  });
  expect(job).toBe('j1');
});
