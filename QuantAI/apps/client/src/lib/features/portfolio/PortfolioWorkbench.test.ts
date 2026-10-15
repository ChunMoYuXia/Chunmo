// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { tick } from 'svelte';
import { configureConnection } from '$lib/api/client';
import type { components } from '$lib/api/schema';
import type { Feature } from '$lib/api/types';
import { researchContext } from '$lib/state/research';
import PortfolioWorkbench from './PortfolioWorkbench.svelte';

vi.mock('$app/state', () => ({ page: { url: new URL('http://localhost/portfolios') } }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
afterEach(() => {
  vi.unstubAllGlobals();
  researchContext.set({ datasetId: '', asOf: '' });
});
const asOf = '2025-01-02T07:00:00Z';
const feature: Feature = {
  security_id: 'XSHE:000001',
  code: '000001',
  name: '测试证券',
  exchange: 'XSHE',
  industry: null,
  close: '10',
  momentum_20d: 0.1,
  volatility_20d: 0.2,
  average_amount_20d: null,
  sample_count: 21,
  last_session: '2025-01-02',
  price_basis: 'raw',
  warnings: [],
  missing_sessions: [],
  stale: false,
};
function portfolio(id: string): components['schemas']['PortfolioDTO'] {
  return {
    id,
    created_at: asOf,
    cash_weight: '0',
    spec: {
      name: id,
      dataset_id: 'd1',
      as_of: asOf,
      target_weights: { 'XSHE:000001': '1' },
      holdings: {},
      cash: '100000',
      max_weight: '1',
      knowledge_mode: 'public_pit',
    },
  };
}
async function setup() {
  configureConnection({ url: '', token: '' });
  researchContext.set({ datasetId: 'd1', asOf });
  let finish: (response: Response) => void = () => {
    throw new Error('no pending command');
  };
  vi.stubGlobal('fetch', (url: string, init: RequestInit) => {
    if (init?.method === 'POST')
      return new Promise<Response>((resolve) => {
        finish = resolve;
      });
    return Promise.resolve(
      Response.json({
        data: url.includes('/snapshot') ? [feature] : [portfolio('p1'), portfolio('p2')],
      }),
    );
  });
  render(PortfolioWorkbench);
  await screen.findByRole('option', { name: 'p1 · 2025-01-02' });
  const user = userEvent.setup();
  await user.selectOptions(screen.getByLabelText('已保存组合'), 'p1');
  await waitFor(() => expect(screen.getByLabelText('XSHE:000001 目标权重 / %')).toHaveValue('100'));
  return { user, finish: (data: unknown) => finish(Response.json({ data })) };
}

it('does not reactivate a saved snapshot after cash is edited during the save', async () => {
  const { user, finish } = await setup();
  await user.click(screen.getByRole('button', { name: '保存组合' }));
  await user.clear(screen.getByLabelText('现金 / 元'));
  await user.type(screen.getByLabelText('现金 / 元'), '200000');
  finish(portfolio('old-save'));
  await tick();
  await waitFor(() => expect(screen.getByRole('button', { name: '保存组合' })).toBeEnabled());
  expect(screen.getByRole('button', { name: '预览调仓' })).toBeDisabled();
  expect(screen.queryByRole('link', { name: '回测此组合 →' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('现金 / 元')).toHaveValue('200000');
});

it('discards a late rebalance plan after weights change', async () => {
  const { user, finish } = await setup();
  await user.click(screen.getByRole('button', { name: '预览调仓' }));
  await user.clear(screen.getByLabelText('XSHE:000001 目标权重 / %'));
  await user.type(screen.getByLabelText('XSHE:000001 目标权重 / %'), '50');
  finish({
    orders: [],
    estimated_fees: '0',
    cash_after: '100000',
    equity_before: '100000',
    fee_schedule: { version: 'test' },
    assumptions: [],
  });
  await tick();
  await waitFor(() => expect(screen.getByRole('button', { name: '保存组合' })).toBeEnabled());
  expect(screen.queryByRole('heading', { name: '整股调仓预览' })).not.toBeInTheDocument();
});

it('keeps a newly loaded portfolio active when an older same-context save finishes', async () => {
  const { user, finish } = await setup();
  await user.click(screen.getByRole('button', { name: '保存组合' }));
  await user.selectOptions(screen.getByLabelText('已保存组合'), 'p2');
  finish(portfolio('old-save'));
  await tick();
  await waitFor(() =>
    expect(screen.getByRole('link', { name: '回测此组合 →' })).toHaveAttribute(
      'href',
      '/backtests?portfolio=p2',
    ),
  );
});
