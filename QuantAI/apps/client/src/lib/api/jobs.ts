import { ApiError, request } from './client';
import type { components } from './schema';

export type JobRecord = components['schemas']['JobRecord'];
export type JobProgress = components['schemas']['Progress'];

function pause(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, 350);
    signal.addEventListener('abort', abort, { once: true });
  });
}

export async function pollJob<T>(
  jobId: string,
  signal: AbortSignal,
  onProgress: (progress: JobProgress) => void,
): Promise<T> {
  const path = `/v1/jobs/${encodeURIComponent(jobId)}`;
  while (true) {
    signal.throwIfAborted();
    const job = await request<JobRecord>(path, { signal });
    onProgress(job.progress);
    if (job.state === 'succeeded') return request<T>(`${path}/result`, { signal });
    if (job.state === 'failed' || job.state === 'cancelled') {
      throw new ApiError(
        job.error?.message ?? '任务已取消',
        job.error?.code ?? 'JOB_CANCELLED',
        job.error?.trace_id ?? job.trace_id ?? '',
        409,
      );
    }
    if (job.state !== 'queued' && job.state !== 'running') {
      throw new ApiError(
        `无法识别任务状态：${String(job.state)}`,
        'UNKNOWN_JOB_STATE',
        job.trace_id,
        502,
      );
    }
    await pause(signal);
  }
}

export function cancelJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
}
