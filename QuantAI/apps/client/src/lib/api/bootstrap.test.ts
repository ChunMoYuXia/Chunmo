// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { connectBackend } from './bootstrap';
afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
  localStorage.clear();
});
it('keeps desktop credentials in memory and waits for the owned authenticated backend', async () => {
  const headers: string[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    headers.push(new Headers(init.headers).get('Authorization') ?? '');
    return Response.json({ data: { mode: 'local' } });
  });
  await connectBackend(
    new URL('http://tauri.localhost/'),
    () => {},
    async () => ({ url: 'http://127.0.0.1:9001', token: 'native-private-test-token' }),
  );
  expect(headers).toEqual(['Bearer native-private-test-token']);
  expect(sessionStorage.length).toBe(0);
  expect(localStorage.length).toBe(0);
});
it('retries an interrupted browser cookie exchange using only in-memory launch credentials', async () => {
  let attempts = 0;
  const headers: string[] = [];
  const cleaned: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    headers.push(new Headers(init.headers).get('Authorization') ?? '');
    if (url.endsWith('/session') && ++attempts === 1) throw new TypeError('lost');
    return Response.json({ data: { ready: true } });
  });
  await expect(
    connectBackend(new URL('http://127.0.0.1:5173/#session=browser-test-token'), (url) =>
      cleaned.push(url),
    ),
  ).rejects.toThrow('lost');
  await connectBackend(new URL(cleaned[0]), () => {});
  expect(headers).toEqual(['Bearer browser-test-token', 'Bearer browser-test-token', '']);
  expect(cleaned[0]).toBe('http://127.0.0.1:5173/');
  expect(sessionStorage.getItem('portfolio-backend-url')).toBe('http://127.0.0.1:8787');
  expect(JSON.stringify({ ...sessionStorage, ...localStorage })).not.toContain(
    'browser-test-token',
  );
});
