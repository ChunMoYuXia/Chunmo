// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import OrderForm from './OrderForm.svelte';

afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});
it('blocks submission during account creation or account reload', async () => {
  render(OrderForm, {
    accountId: 'a1',
    paused: false,
    disabled: true,
    securities: [{ security_id: 'XSHG:600001', name: '示例证券' }],
    onChange: () => {},
  });
  expect(screen.getByRole('button', { name: '提交模拟订单' })).toBeDisabled();
});
it('keeps the same key on a conflict retry and creates a new key only for changed content', async () => {
  configureConnection({ url: '', token: '' });
  const keys: string[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    keys.push(new Headers(init.headers).get('Idempotency-Key')!);
    return Response.json(
      {
        error: {
          code: 'IDEMPOTENCY_CONFLICT',
          message: '幂等键内容冲突',
          trace_id: 'conflict-trace',
          details: {},
        },
      },
      { status: 409 },
    );
  });
  render(OrderForm, {
    accountId: 'a1',
    paused: false,
    securities: [{ security_id: '600001.SH', name: '示例证券' }],
    onChange: () => {},
  });
  const user = userEvent.setup();
  await user.click(screen.getByRole('button', { name: '提交模拟订单' }));
  expect(await screen.findByText('幂等键内容冲突')).toBeVisible();
  expect(screen.getByText(/conflict-trace/)).toBeVisible();
  await user.click(screen.getByRole('button', { name: '重试同一订单' }));
  await screen.findByText('幂等键内容冲突');
  expect(keys).toHaveLength(2);
  expect(keys[1]).toBe(keys[0]);
  await user.clear(screen.getByLabelText('委托数量 / 股'));
  await user.type(screen.getByLabelText('委托数量 / 股'), '200');
  await user.click(screen.getByRole('button', { name: '重试同一订单' }));
  await screen.findByText('幂等键内容冲突');
  expect(keys).toHaveLength(3);
  expect(keys[2]).not.toBe(keys[0]);
});
