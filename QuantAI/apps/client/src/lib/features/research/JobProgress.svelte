<script lang="ts">
  import { pollJob, cancelJob, type JobProgress as Progress } from '$lib/api/jobs';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let {
    jobId,
    onResult,
  }: { jobId: string; onResult: (value: components['schemas']['ResearchArtifact']) => void } =
    $props();
  let progress = $state<Progress>({ completed: 0, total: 0, message: '等待任务' }),
    failure = $state<unknown>(null),
    jobState = $state('running'),
    busy = $state(false),
    retry = $state(0);
  let controller: AbortController;
  $effect(() => {
    const id = jobId;
    void retry;
    controller = new AbortController();
    const active = controller;
    failure = null;
    jobState = 'running';
    pollJob<components['schemas']['ResearchArtifact']>(id, active.signal, (value) => {
      if (!active.signal.aborted) progress = value;
    })
      .then((result) => {
        if (!active.signal.aborted) {
          jobState = 'succeeded';
          onResult(result);
        }
      })
      .catch((error) => {
        if (!active.signal.aborted) {
          failure = error;
          jobState = error?.code === 'JOB_CANCELLED' ? 'cancelled' : 'failed';
        }
      });
    return () => active.abort();
  });
  async function cancel() {
    busy = true;
    try {
      const job = await cancelJob(jobId);
      if (job.state === 'cancelled') {
        controller.abort();
        jobState = 'cancelled';
        failure = null;
      } else if (job.state === 'succeeded') {
        retry++;
      }
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel" aria-label="研究任务">
  <div class="section-heading">
    <h2>研究任务</h2>
    <code>{jobId}</code>
  </div>
  <p>{progress.message} · {progress.completed} / {progress.total}</p>
  <progress value={progress.completed} max={progress.total || 1}></progress>
  {#if jobState === 'running'}<button disabled={busy} onclick={() => void cancel()}
      >取消后台任务</button
    >{:else if jobState === 'cancelled'}<p role="status">
      任务已取消
    </p>{:else if jobState === 'succeeded'}<p role="status">研究结果已保存</p>{/if}
  {#if failure}<StatusPanel title="任务状态异常" error={failure} /><button onclick={() => retry++}
      >重新查询任务</button
    >{/if}
  <p class="context-note">离开页面只停止查询；取消按钮会请求后台终止任务。</p>
</section>
