import { afterEach, expect, it, vi } from 'vitest';
import { configureConnection } from './client';
import { cancelJob, pollJob } from './jobs';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

it('polls progress and returns persisted results without submitting a duplicate task', async () => {
  vi.useFakeTimers();
  configureConnection({ url: '', token: '' });
  const requests: string[] = [];
  let polls = 0;
  vi.stubGlobal('fetch', async (url: string) => {
    requests.push(url);
    return Response.json({
      data: url.endsWith('/result')
        ? { nav: [100, 110] }
        : {
            id: 'job-1',
            state: ++polls === 1 ? 'running' : 'succeeded',
            progress: { completed: polls, total: 2, message: '研究' },
          },
    });
  });
  const progress = vi.fn();
  const pending = pollJob<{ nav: number[] }>('job-1', new AbortController().signal, progress);
  await vi.runAllTimersAsync();
  expect(await pending).toEqual({ nav: [100, 110] });
  expect(progress.mock.calls[0][0].completed).toBe(1);
  expect(requests).toEqual(['/v1/jobs/job-1', '/v1/jobs/job-1', '/v1/jobs/job-1/result']);
});

it('stopping UI polling does not cancel the server job; cancellation is explicit', async () => {
  configureConnection({ url: '', token: '' });
  const methods: string[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    methods.push(init.method ?? 'GET');
    return Response.json({
      data: { id: 'job-1', state: 'cancelled', progress: { completed: 0, total: 2, message: '' } },
    });
  });
  const controller = new AbortController();
  controller.abort();
  await expect(pollJob('job-1', controller.signal, () => {})).rejects.toMatchObject({
    name: 'AbortError',
  });
  expect(methods).toEqual([]);
  expect((await cancelJob('job-1')).state).toBe('cancelled');
  expect(methods).toEqual(['POST']);
});
