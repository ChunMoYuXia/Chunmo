// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import HaltButton from './HaltButton.svelte';

afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});

it('halts immediately and requires the recovery phrase only when resuming', async () => {
  configureConnection({ url: '', token: '' });
  const calls: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    calls.push(url);
    expect(new Headers(init.headers).get('Idempotency-Key')).toBeTruthy();
    return Response.json({
      data: url.endsWith('/halt')
        ? { account: { status: 'paused' }, canceled_order_ids: [], unknown_order_ids: [] }
        : { status: 'active' },
    });
  });
  render(HaltButton, { accountId: 'a1', paused: false, onChange: () => {} });
  const user = userEvent.setup();
  await user.click(screen.getByRole('button', { name: '紧急停止' }));
  expect(await screen.findByText('已暂停新订单')).toBeVisible();
  expect(screen.queryByRole('dialog')).toBeNull();
  await user.click(screen.getByRole('button', { name: '恢复模拟交易' }));
  const confirm = screen.getByRole('button', { name: '确认恢复' });
  expect(confirm).toBeDisabled();
  expect(calls).toEqual(['/v1/paper/accounts/a1/halt']);
  await user.type(screen.getByLabelText('恢复确认短语'), '恢复模拟交易');
  await user.click(confirm);
  expect(await screen.findByText('已恢复模拟交易')).toBeVisible();
});
