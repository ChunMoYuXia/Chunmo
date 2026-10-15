import { ApiError, openStream } from './client';
import type { components } from './schema';
export type AIEvent = components['schemas']['AIEvent'];
const eventTypes = new Set([
  'started',
  'tool_result',
  'paragraph',
  'usage',
  'succeeded',
  'failed',
  'cancelled',
]);
const terminalTypes = new Set(['succeeded', 'failed', 'cancelled']);

/** Recover events from the existing run. This endpoint never initiates model inference. */
export async function streamAIEvents(
  runId: string,
  after: number,
  signal: AbortSignal,
  onEvent: (event: AIEvent) => void,
): Promise<void> {
  const response = await openStream(`/v1/ai/runs/${encodeURIComponent(runId)}/events`, {
    method: 'POST',
    body: JSON.stringify({ after }),
    signal,
  });
  const reader = response.body?.getReader();
  if (!reader) throw new ApiError('事件流为空', 'INVALID_STREAM', '', 502);
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '',
    cursor = after,
    size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 1_000_000)
        throw new ApiError('事件流超出本地大小限制', 'AI_STREAM_BUDGET', '', 502);
      buffer += decoder.decode(value, { stream: true });
      // Keep incomplete CRLF pairs intact across network chunks.
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary: number;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const raw = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (!raw) continue;
        let event: AIEvent;
        try {
          event = JSON.parse(raw);
        } catch {
          throw new ApiError('事件格式无效', 'INVALID_AI_EVENT', '', 502);
        }
        if (
          event?.run_id !== runId ||
          event.version !== 1 ||
          !Number.isSafeInteger(event.seq) ||
          event.seq < 1 ||
          !eventTypes.has(event.type) ||
          !event.payload ||
          typeof event.payload !== 'object'
        )
          throw new ApiError('事件版本或内容无法识别', 'INVALID_AI_EVENT', '', 502);
        if (event.seq <= cursor) continue;
        if (event.seq !== cursor + 1)
          throw new ApiError('事件序列不连续，请查询原运行记录', 'AI_EVENT_GAP', '', 502);
        cursor = event.seq;
        onEvent(event);
        if (terminalTypes.has(event.type)) return;
      }
    }
    throw new ApiError('事件连接已断开；可继续查询原运行', 'AI_STREAM_DISCONNECTED', '', 0);
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
