import { afterEach, expect, it, vi } from 'vitest';
import { configureConnection } from './client';
import { streamAIEvents } from './ai-events';
afterEach(() => vi.unstubAllGlobals());
it('decodes split UTF-8 frames, ignores replay duplicates and requires a terminal event', async () => {
  configureConnection({ url: '', token: '' });
  const paragraph = {
    run_id: 'r1',
    seq: 1,
    version: 1,
    type: 'paragraph',
    payload: { message: '', text: '已校验', evidence_ids: ['e1'] },
  };
  const done = { ...paragraph, seq: 2, type: 'succeeded' };
  const bytes = new TextEncoder().encode(
    `data: ${JSON.stringify(paragraph)}\r\n\r\ndata: ${JSON.stringify(paragraph)}\n\ndata: ${JSON.stringify(done)}\n\n`,
  );
  vi.stubGlobal(
    'fetch',
    async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3));
            controller.close();
          },
        }),
        { headers: { 'Content-Type': 'text/event-stream' } },
      ),
  );
  const seen: number[] = [];
  await streamAIEvents('r1', 0, new AbortController().signal, (event) => seen.push(event.seq));
  expect(seen).toEqual([1, 2]);
});
it('treats premature closure as disconnected instead of declaring completion', async () => {
  configureConnection({ url: '', token: '' });
  vi.stubGlobal(
    'fetch',
    async () =>
      new Response(
        'data: {"run_id":"r1","seq":1,"version":1,"type":"started","payload":{"message":"开始"}}\n\n',
        { headers: { 'Content-Type': 'text/event-stream' } },
      ),
  );
  await expect(
    streamAIEvents('r1', 0, new AbortController().signal, () => {}),
  ).rejects.toMatchObject({ code: 'AI_STREAM_DISCONNECTED' });
});
