import { request } from './client';

type Pending = { path: string; body: string; key: string };

/** Explicit user commands only. Keep the attempt on uncertain failures and reload. */
export class CommandClient {
  private pending: Pending | null = null;
  constructor(private readonly slot: string) {
    try {
      const saved: unknown = JSON.parse(sessionStorage.getItem(this.storageKey) ?? 'null');
      if (
        saved &&
        typeof saved === 'object' &&
        'path' in saved &&
        'body' in saved &&
        'key' in saved &&
        typeof saved.path === 'string' &&
        typeof saved.body === 'string' &&
        typeof saved.key === 'string'
      )
        this.pending = saved as Pending;
    } catch {
      /* Storage is optional; credentials are never written here. */
    }
  }
  private get storageKey() {
    return `pending-command-v1:${this.slot}`;
  }
  async send<T>(path: string, payload: unknown): Promise<T> {
    const body = JSON.stringify(payload);
    if (this.pending?.path !== path || this.pending.body !== body)
      this.pending = { path, body, key: crypto.randomUUID() };
    const attempt = this.pending;
    try {
      sessionStorage.setItem(this.storageKey, JSON.stringify(attempt));
    } catch {
      /* optional */
    }
    const response = await request<T>(path, {
      method: 'POST',
      body,
      headers: { 'Idempotency-Key': attempt.key },
    });
    if (this.pending === attempt) {
      this.pending = null;
      try {
        sessionStorage.removeItem(this.storageKey);
      } catch {
        /* optional */
      }
    }
    return response;
  }
}
