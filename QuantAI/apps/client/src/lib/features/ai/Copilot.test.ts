// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { configureConnection } from '$lib/api/client';
import Copilot from './Copilot.svelte';
import userEvent from '@testing-library/user-event';
afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});
it('shows the actual unconfigured state without fabricated model content', async () => {
  configureConnection({ url: '', token: '' });
  vi.stubGlobal('fetch', async () =>
    Response.json({
      data: {
        configured: false,
        model: null,
        max_output_tokens: 2048,
        total_timeout_seconds: 60,
        usage_note: '显式发起才调用模型',
      },
    }),
  );
  render(Copilot, { reportId: 'report1' });
  expect(await screen.findByText('AI 尚未配置')).toBeVisible();
  expect(screen.getByRole('button', { name: '发起 AI 解释' })).toBeDisabled();
  expect(screen.queryByLabelText('AI 已验证解释')).toBeNull();
});

it('recovers the original run after a disconnect without starting paid inference again', async () => {
  configureConnection({ url: '', token: '' });
  let submissions = 0;
  const initial = {
    id: 'r1',
    state: 'running',
    model: 'example',
    events: [],
    request: { mode: 'explain', report_id: 'report1', question: '解释' },
    trace_id: 't1',
  };
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('/config'))
      return Response.json({ data: { configured: true, model: 'example' } });
    if (url.endsWith('/events')) throw new TypeError('disconnected');
    if (url.endsWith('/runs') && init.method === 'POST') {
      submissions++;
      return Response.json({ data: initial });
    }
    return Response.json({
      data: {
        ...initial,
        state: 'succeeded',
        events: [
          {
            run_id: 'r1',
            seq: 1,
            version: 1,
            type: 'paragraph',
            payload: { text: '已校验的研究解释。', evidence_ids: ['report1'], message: '' },
          },
          { run_id: 'r1', seq: 2, version: 1, type: 'succeeded', payload: { message: '完成' } },
        ],
      },
    });
  });
  render(Copilot, { reportId: 'report1' });
  const start = await screen.findByRole('button', { name: '发起 AI 解释' });
  await userEvent.click(start);
  await userEvent.click(await screen.findByRole('button', { name: '继续查询原运行' }));
  expect(await screen.findByText('已校验的研究解释。')).toBeVisible();
  expect(submissions).toBe(1);
});

it('discards late recovery results after selecting another report', async () => {
  configureConnection({ url: '', token: '' });
  let finish: (value: Response) => void = () => {};
  const initial = {
    id: 'r1',
    state: 'running',
    model: 'example',
    events: [],
    request: { mode: 'explain', report_id: 'report1', question: '解释' },
    trace_id: 't1',
  };
  vi.stubGlobal('fetch', async (url: string) => {
    if (url.endsWith('/config'))
      return Response.json({ data: { configured: true, model: 'example' } });
    if (url.endsWith('/events')) throw new TypeError('disconnected');
    if (url.endsWith('/runs')) return Response.json({ data: initial });
    return new Promise<Response>((resolve) => (finish = resolve));
  });
  const view = render(Copilot, { reportId: 'report1' });
  await userEvent.click(await screen.findByRole('button', { name: '发起 AI 解释' }));
  await userEvent.click(await screen.findByRole('button', { name: '继续查询原运行' }));
  await view.rerender({ reportId: 'report2' });
  finish(
    Response.json({
      data: {
        ...initial,
        state: 'succeeded',
        events: [
          {
            run_id: 'r1',
            seq: 1,
            version: 1,
            type: 'paragraph',
            payload: { text: '旧报告的解释。', message: '', evidence_ids: ['report1'] },
          },
        ],
      },
    }),
  );
  await new Promise((resolve) => setTimeout(resolve, 10));
  expect(screen.queryByText('旧报告的解释。')).toBeNull();
});
