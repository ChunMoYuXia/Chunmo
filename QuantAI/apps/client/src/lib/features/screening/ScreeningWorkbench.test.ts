// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { tick } from 'svelte';
import { researchContext } from '$lib/state/research';
import type { Feature } from '$lib/api/types';
import ScreeningWorkbench from './ScreeningWorkbench.svelte';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
afterEach(() => {
  vi.unstubAllGlobals();
  researchContext.set({ datasetId: '', asOf: '' });
});

function feature(code: string, name: string, close: string | null): Feature {
  return {
    security_id: `XSHE:${code}`,
    code,
    name,
    close,
    exchange: 'XSHE',
    industry: '科技',
    momentum_20d: 0.2,
    volatility_20d: 0.1,
    average_amount_20d: '100000',
    sample_count: 21,
    last_session: '2024-01-02',
    price_basis: 'raw',
    warnings: [],
    missing_sessions: [],
    stale: false,
  };
}

it('runs against the original selected pool and recomputes a shorter prefix on rollback', async () => {
  const rows = [
    feature('000001', '股票甲', '8'),
    feature('000002', '股票乙', null),
    feature('000003', '股票丙', '12'),
  ];
  researchContext.set({ datasetId: 'd1', asOf: '2024-01-02T07:00:00Z' });
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    let data: unknown = [];
    if (url.includes('/snapshot')) data = rows;
    if (url.endsWith('/fields'))
      data = [
        {
          key: 'close',
          label: '收盘价',
          data_type: 'number',
          unit: 'CNY',
          display_unit: '元',
          display_scale: 1,
        },
        {
          key: 'momentum_20d',
          label: '20日动量',
          data_type: 'number',
          unit: 'ratio',
          display_unit: '%',
          display_scale: 100,
        },
      ];
    if (url === '/v1/screening') {
      const body = JSON.parse(String(init.body));
      data = {
        universe_count: body.universe.length,
        total_matching_count: body.filters.length ? 1 : 3,
        rows: body.filters.length ? [rows[2]] : rows,
        steps: body.filters.length
          ? [
              {
                index: 0,
                input_count: 3,
                remaining_count: 1,
                condition: body.filters[0],
                exclusions: { missing_value: 1, below_threshold: 1 },
              },
            ]
          : [],
      };
    }
    return Response.json({ data });
  });
  render(ScreeningWorkbench);
  const user = userEvent.setup();
  await screen.findByRole('checkbox', { name: '纳入 股票甲' });
  await user.click(screen.getByRole('button', { name: '添加条件' }));
  await user.selectOptions(screen.getByLabelText('指标 1'), 'close');
  const threshold = screen.getByLabelText('条件值 1');
  await user.clear(threshold);
  await user.type(threshold, '10');
  await user.click(screen.getByRole('button', { name: '运行筛选' }));
  expect(await screen.findByText('数据缺失：1')).toBeVisible();
  expect(within(screen.getByRole('region', { name: '筛选结果' })).getAllByRole('row')).toHaveLength(
    2,
  );
  await user.click(screen.getByRole('button', { name: '撤回条件 1' }));
  await screen.findByText('未添加条件，保留原始候选池。');
  expect(within(screen.getByRole('region', { name: '筛选结果' })).getAllByRole('row')).toHaveLength(
    4,
  );
});

it('clears an old running state and rejects its late result after changing datasets', async () => {
  const rows = [feature('000001', '股票甲', '10')];
  let finishOld: (value: Response) => void = () => {};
  researchContext.set({ datasetId: 'd1', asOf: '2024-01-02T07:00:00Z' });
  vi.stubGlobal('fetch', (url: string) => {
    if (url === '/v1/screening') return new Promise<Response>((resolve) => (finishOld = resolve));
    const data = url.includes('/snapshot')
      ? rows
      : url.endsWith('/fields')
        ? [
            {
              key: 'momentum_20d',
              label: '20日动量',
              data_type: 'number',
              unit: 'ratio',
              display_unit: '%',
              display_scale: 100,
            },
          ]
        : [];
    return Promise.resolve(Response.json({ data }));
  });
  render(ScreeningWorkbench);
  await screen.findByRole('checkbox', { name: '纳入 股票甲' });
  await userEvent.click(screen.getByRole('button', { name: '运行筛选' }));
  expect(screen.getByRole('button', { name: '正在筛选…' })).toBeDisabled();
  researchContext.set({ datasetId: 'd2', asOf: '2024-01-02T07:00:00Z' });
  await tick();
  expect(await screen.findByRole('button', { name: '运行筛选' })).toBeEnabled();
  finishOld(
    Response.json({ data: { universe_count: 1, total_matching_count: 1, rows, steps: [] } }),
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(screen.queryByRole('region', { name: '筛选结果' })).not.toBeInTheDocument();
});
