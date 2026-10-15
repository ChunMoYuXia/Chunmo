// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import OptimizationForm from './OptimizationForm.svelte';
afterEach(() => vi.unstubAllGlobals());
it('explains infeasible caps without replacing draft weights', async () => {
  configureConnection({ url: '', token: '' });
  let changed = false;
  vi.stubGlobal('fetch', async () =>
    Response.json(
      {
        error: {
          code: 'INFEASIBLE_CONSTRAINTS',
          message: '个股上限不足以分配全部资产',
          trace_id: 'cap-trace',
          details: {},
        },
      },
      { status: 422 },
    ),
  );
  render(OptimizationForm, {
    props: {
      context: { datasetId: 'd1', asOf: '2025-01-02T07:00:00Z' },
      universe: ['600001.SH'],
      maxWeight: '0.2',
      onResult: () => {
        changed = true;
      },
    },
  });
  await userEvent.click(screen.getByRole('button', { name: '计算目标权重' }));
  expect(await screen.findByText('个股上限不足以分配全部资产')).toBeVisible();
  expect(changed).toBe(false);
});
