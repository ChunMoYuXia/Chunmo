import type { components } from './schema';

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly traceId: string,
    public readonly status: number,
    public readonly details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

export type BackendConnection = { url: string; token: string };

let connection: BackendConnection = { url: '', token: '' };

function validateBackendUrl(value: string): void {
  if (value) {
    const url = new URL(value);
    if (
      url.protocol !== 'http:' ||
      !['127.0.0.1', 'localhost'].includes(url.hostname) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      url.pathname !== '/'
    ) {
      throw new ApiError('只允许连接本机研究服务', 'INVALID_SERVER', '', 0);
    }
  }
}

export function configureConnection(next: BackendConnection): void {
  validateBackendUrl(next.url);
  connection = { url: next.url.replace(/\/$/, ''), token: next.token };
}

export function browserBackendUrl(): string {
  try {
    const value = sessionStorage.getItem('portfolio-backend-url');
    if (value) {
      validateBackendUrl(value);
      return value;
    }
  } catch {
    /* A browser may disable storage. */
  }
  return 'http://127.0.0.1:8787';
}

export function rememberBackendUrl(url: string): void {
  validateBackendUrl(url);
  try {
    sessionStorage.setItem('portfolio-backend-url', url);
  } catch {
    /* Keep credentials in memory only. */
  }
}

export function consumeSessionLink(
  input: URL,
): { connection: BackendConnection; cleanUrl: string } | null {
  const fragment = new URLSearchParams(input.hash.slice(1));
  const token = fragment.get('session');
  if (!token) return null;
  const clean = new URL(input);
  clean.hash = '';
  return {
    connection: { url: fragment.get('backend') ?? 'http://127.0.0.1:8787', token },
    cleanUrl: clean.href,
  };
}

async function fetchFromBackend(
  path: string,
  init: RequestInit,
  accept: string,
): Promise<Response> {
  if (!path.startsWith('/v1/') && path !== '/health') {
    throw new ApiError('请求路径不受支持', 'INVALID_PATH', '', 0);
  }
  const headers = new Headers(init.headers);
  headers.set('Accept', accept);
  if (typeof init.body === 'string') headers.set('Content-Type', 'application/json');
  if (connection.token) headers.set('Authorization', `Bearer ${connection.token}`);

  return fetch(`${connection.url}${path}`, {
    ...init,
    headers,
    credentials: 'include',
    cache: 'no-store',
  });
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return decodeResponse<T>(await fetchFromBackend(path, init, 'application/json'));
}

export async function openStream(path: string, init: RequestInit): Promise<Response> {
  const response = await fetchFromBackend(path, init, 'text/event-stream');
  if (!response.ok) await decodeResponse(response);
  if (!response.headers.get('Content-Type')?.startsWith('text/event-stream'))
    throw new ApiError(
      '本地服务返回了不兼容的事件流',
      'INVALID_STREAM',
      response.headers.get('X-Trace-Id') ?? '',
      response.status,
    );
  return response;
}

async function decodeResponse<T>(response: Response): Promise<T> {
  let body: { data?: T; error?: components['schemas']['ErrorBody'] };
  try {
    body = await response.json();
    if (body === null || typeof body !== 'object' || Array.isArray(body))
      throw new Error('invalid envelope');
  } catch {
    throw new ApiError(
      '本地服务返回了无效响应',
      'INVALID_RESPONSE',
      response.headers.get('X-Trace-Id') ?? '',
      response.status,
    );
  }
  if (!response.ok) {
    throw new ApiError(
      body.error?.message ?? '请求失败',
      body.error?.code ?? 'REQUEST_FAILED',
      body.error?.trace_id ?? response.headers.get('X-Trace-Id') ?? '',
      response.status,
      body.error?.details ?? {},
    );
  }
  if (!Object.hasOwn(body, 'data')) {
    throw new ApiError('本地服务返回了无效响应', 'INVALID_RESPONSE', '', response.status);
  }
  return body.data as T;
}
