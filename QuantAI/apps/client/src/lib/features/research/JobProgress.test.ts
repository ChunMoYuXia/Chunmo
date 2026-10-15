// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import JobProgress from './JobProgress.svelte';

afterEach(() => vi.unstubAllGlobals());
it('surfaces an unknown server state instead of polling forever', async () => {
  configureConnection({ url: '', token: '' });
  vi.stubGlobal('fetch', async () =>
    Response.json({
      data: {
        id: 'j1',
        state: 'unexpected',
        trace_id: 'unknown-trace',
        progress: { completed: 1, total: 2, message: '未知' },
      },
    }),
  );
  render(JobProgress, { jobId: 'j1', onResult: () => {} });
  expect(await screen.findByText(/UNKNOWN_JOB_STATE/)).toBeVisible();
  expect(screen.getByText(/unknown-trace/)).toBeVisible();
});
it('cancels explicitly while preserving visible progress and terminal status', async () => {
  configureConnection({ url: '', token: '' });
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) =>
    Response.json({
      data: {
        id: 'j1',
        state: init.method === 'POST' ? 'cancelled' : 'running',
        progress: { completed: 15, total: 30, message: '计算中' },
        trace_id: 't1',
      },
    }),
  );
  render(JobProgress, { jobId: 'j1', onResult: () => {} });
  expect(await screen.findByText(/15 \/ 30/)).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: '取消后台任务' }));
  expect(await screen.findByText('任务已取消')).toBeVisible();
  expect(screen.queryByRole('button', { name: '取消后台任务' })).toBeNull();
});
