import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  configureConnection,
  consumeSessionLink,
  request,
  browserBackendUrl,
  rememberBackendUrl,
} from './client';

afterEach(() => vi.unstubAllGlobals());

it.each([null, [], 'unexpected'])(
  'rejects a non-envelope JSON response without leaking a runtime exception: %s',
  async (body) => {
    configureConnection({ url: '', token: '' });
    vi.stubGlobal('fetch', async () => Response.json(body));
    await expect(request('/v1/config')).rejects.toMatchObject({ code: 'INVALID_RESPONSE' });
  },
);

describe('API transport', () => {
  it('remembers only the backend address for browser reload on a custom port', () => {
    const storage = new Map<string, string>();
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    });
    rememberBackendUrl('http://127.0.0.1:9000');
    expect(browserBackendUrl()).toBe('http://127.0.0.1:9000');
    expect([...storage.values()]).toEqual(['http://127.0.0.1:9000']);
    expect(() => rememberBackendUrl('https://external.example')).toThrow(ApiError);
  });
  it('keeps field-level validation evidence for the input form', async () => {
    vi.stubGlobal('fetch', async () =>
      Response.json(
        {
          error: {
            code: 'VALIDATION_ERROR',
            message: '输入不合法',
            trace_id: 'trace-fields',
            details: { issues: [{ location: ['body', 'quantity'], message: '必须为整数' }] },
          },
        },
        { status: 422 },
      ),
    );
    await expect(request('/v1/paper/accounts')).rejects.toMatchObject({
      details: { issues: [{ location: ['body', 'quantity'], message: '必须为整数' }] },
    });
  });
  it('returns data and attaches the current session', async () => {
    configureConnection({ url: 'http://127.0.0.1:8787', token: 'test-session' });
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
      if (new Headers(init.headers).get('Authorization') !== 'Bearer test-session') {
        return new Response('Unauthorized', { status: 401 });
      }
      return Response.json({ data: { count: 3 }, meta: { trace_id: 'trace-1' } });
    });
    expect(await request('/v1/config')).toEqual({ count: 3 });
  });

  it('preserves the actionable code and trace on failed writes', async () => {
    vi.stubGlobal('fetch', async () =>
      Response.json(
        {
          error: {
            code: 'IDEMPOTENCY_CONFLICT',
            message: '订单内容不同',
            trace_id: 'trace-2',
            details: {},
          },
        },
        { status: 409 },
      ),
    );
    try {
      await request('/v1/orders', { method: 'POST' });
      throw new Error('Expected request failure');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).code).toBe('IDEMPOTENCY_CONFLICT');
      expect((error as ApiError).traceId).toBe('trace-2');
    }
  });

  it('does not send the session to an external server', () => {
    expect(() =>
      configureConnection({ url: 'https://untrusted.example', token: 'private' }),
    ).toThrow(ApiError);
  });

  it('consumes launch credentials without preserving them in the URL', () => {
    const link = consumeSessionLink(
      new URL('http://127.0.0.1:5173/data#session=private&backend=http%3A%2F%2F127.0.0.1%3A8787'),
    );
    expect(link?.connection).toEqual({ url: 'http://127.0.0.1:8787', token: 'private' });
    expect(link?.cleanUrl).toBe('http://127.0.0.1:5173/data');
  });
});
