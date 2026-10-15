// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { CommandClient } from './commands';
import { configureConnection } from './client';
afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});
it('recovers an uncertain command after reload without replaying it automatically', async () => {
  configureConnection({ url: '', token: '' });
  const keys: string[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    keys.push(new Headers(init.headers).get('Idempotency-Key')!);
    if (keys.length === 1) throw new TypeError('connection lost');
    return Response.json({ data: { id: 'order1' } });
  });
  await expect(
    new CommandClient('order').send('/v1/paper/accounts/a/orders', { quantity: 100 }),
  ).rejects.toThrow('connection lost');
  const afterReload = new CommandClient('order');
  expect(keys).toHaveLength(1);
  expect(await afterReload.send('/v1/paper/accounts/a/orders', { quantity: 100 })).toEqual({
    id: 'order1',
  });
  expect(keys[1]).toBe(keys[0]);
});
