<script lang="ts">
  import { untrack } from 'svelte';
  import { request } from '$lib/api/client';
  import { CommandClient } from '$lib/api/commands';
  import { RequestScope } from '$lib/api/scope';
  import { streamAIEvents, type AIEvent } from '$lib/api/ai-events';
  import type { components } from '$lib/api/schema';
  import type { ScreeningSpec } from '$lib/api/types';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let {
    reportId,
    scope,
    onSpec,
  }: { reportId?: string; scope?: ScreeningSpec; onSpec?: (spec: ScreeningSpec) => void } =
    $props();
  let config = $state<components['schemas']['AIConfig'] | null>(null),
    run = $state<components['schemas']['AIRun'] | null>(null),
    events = $state<AIEvent[]>([]),
    question = $state('解释研究结果的主要风险和数据缺口。'),
    budget = $state(1024),
    busy = $state(false),
    failure = $state<unknown>(null),
    disconnected = $state(false);
  const requests = new RequestScope();
  const command = new CommandClient('ai-start');
  let controller: AbortController | undefined;
  let contextKey = $derived(
    reportId
      ? `report:${reportId}`
      : scope
        ? `screen:${scope.dataset_id}:${scope.as_of}:${scope.knowledge_mode ?? 'public_pit'}:${scope.universe.join(',')}`
        : '',
  );
  let storageKey = $derived(`ai-run-v1:${contextKey}`);
  let paragraphs = $derived(events.filter((event) => event.type === 'paragraph'));
  let status = $derived(events.at(-1));
  $effect(() => {
    const key = contextKey;
    const ticket = requests.start();
    controller?.abort();
    config = null;
    run = null;
    events = [];
    failure = null;
    busy = false;
    disconnected = false;
    if (!key) return () => requests.cancel();
    untrack(() => {
      question = reportId
        ? '解释研究结果的主要风险和数据缺口。'
        : '筛选动量较强且行情未过期的证券。';
    });
    request<components['schemas']['AIConfig']>('/v1/ai/config', { signal: ticket.signal })
      .then((value) => {
        if (ticket.current()) config = value;
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      });
    let previous: string | null = null;
    try {
      previous = sessionStorage.getItem(`ai-run-v1:${key}`);
    } catch {
      /* optional */
    }
    if (previous)
      request<components['schemas']['AIRun']>(`/v1/ai/runs/${encodeURIComponent(previous)}`, {
        signal: ticket.signal,
      })
        .then((value) => {
          if (ticket.current()) {
            run = value;
            events = value.events;
            disconnected = value.state === 'running';
          }
        })
        .catch((error) => {
          if (ticket.current()) failure = error;
        });
    return () => {
      requests.cancel();
      controller?.abort();
    };
  });
  async function listen(id: string) {
    controller?.abort();
    controller = new AbortController();
    const active = controller;
    busy = true;
    disconnected = false;
    try {
      await streamAIEvents(id, events.at(-1)?.seq ?? 0, active.signal, (event) => {
        if (active.signal.aborted) return;
        events = [...events, event];
        if (run && ['succeeded', 'failed', 'cancelled'].includes(event.type))
          run = { ...run, state: event.type as 'succeeded' | 'failed' | 'cancelled' };
      });
    } catch (error) {
      if (!active.signal.aborted) {
        failure = error;
        disconnected = true;
      }
    } finally {
      if (!active.signal.aborted) busy = false;
    }
  }
  async function start() {
    busy = true;
    failure = null;
    disconnected = false;
    const key = storageKey;
    try {
      const value = await command.send<components['schemas']['AIRun']>('/v1/ai/runs', {
        mode: reportId ? 'explain' : 'screen',
        report_id: reportId ?? null,
        scope: scope ?? null,
        question,
        max_output_tokens: budget,
      });
      try {
        sessionStorage.setItem(key, value.id);
      } catch {
        /* optional */
      }
      if (key !== storageKey) return;
      run = value;
      events = [];
      await listen(value.id);
    } catch (error) {
      if (key === storageKey) {
        failure = error;
        busy = false;
      }
    }
  }
  async function recover() {
    if (!run) return;
    const key = contextKey;
    const id = run.id;
    failure = null;
    try {
      const value = await request<components['schemas']['AIRun']>(
        `/v1/ai/runs/${encodeURIComponent(id)}`,
      );
      if (key !== contextKey) return;
      run = value;
      events = value.events;
      disconnected = false;
      if (value.state === 'running') await listen(value.id);
    } catch (error) {
      if (key === contextKey) {
        failure = error;
        disconnected = true;
      }
    }
  }
  async function cancel() {
    if (!run) return;
    const key = contextKey;
    const id = run.id;
    failure = null;
    try {
      const value = await request<components['schemas']['AIRun']>(
        `/v1/ai/runs/${encodeURIComponent(id)}/cancel`,
        { method: 'POST' },
      );
      if (key !== contextKey) return;
      controller?.abort();
      run = value;
      events = value.events;
      busy = false;
      disconnected = false;
    } catch (error) {
      if (key === contextKey) failure = error;
    }
  }
</script>

<section class="panel">
  <div class="section-heading">
    <h2>{reportId ? 'AI 研究解释' : '自然语言选股'}</h2>
    <span class="badge">可选 · 只读工具</span>
  </div>
  {#if config && !config.configured}<StatusPanel
      title="AI 尚未配置"
      message="规则报告与手动选股可独立使用。按设置页说明在本地后端配置模型端点、模型名称和密钥。"
    />{/if}
  <p class="context-note">
    {config?.model ?? '未选择模型'} · 数值只引用后端指标；生成的选股条件需手动应用。
  </p>
  <label
    >研究问题<textarea bind:value={question} maxlength="1000" rows="3" disabled={busy}
    ></textarea></label
  ><label
    >输出 token 上限<select bind:value={budget} disabled={busy}
      ><option value={512}>512</option><option value={1024}>1024</option><option value={2048}
        >2048</option
      ></select
    ></label
  >
  <p class="context-note">
    发起后会向已配置服务发送问题与只读研究上下文。取消或断线时用量可能未知，费用以服务商账单为准。
  </p>
  <div class="actions">
    <button
      class="primary"
      disabled={!config?.configured || busy || !question.trim() || run?.state === 'running'}
      onclick={() => void start()}>{reportId ? '发起 AI 解释' : '生成选股条件'}</button
    >{#if run?.state === 'running'}<button onclick={() => void cancel()}>取消 AI 运行</button
      >{/if}{#if disconnected}<button onclick={() => void recover()}>继续查询原运行</button>{/if}
  </div>
  {#if run}<p class="context-note">
      运行 {run.id} · 模型 {run.model} · {run.state}。离开页面只停止读取事件。
    </p>{/if}
  {#if status}<p role="status" class:warning={status.type === 'failed'}>
      {status.payload.message}
      {status.payload.code ?? ''}
    </p>{/if}
  {#if paragraphs.length}<div aria-label="AI 已验证解释">
      {#each paragraphs as paragraph (paragraph.seq)}<article>
          <p style="white-space:pre-line">{paragraph.payload.text}</p>
          <small>证据：{paragraph.payload.evidence_ids?.join('、')}</small
          >{#if paragraph.payload.spec}<pre class="dsl-preview">{JSON.stringify(
                paragraph.payload.spec.filters,
                null,
                2,
              )}</pre>
            {#if onSpec}<button
                onclick={() => {
                  if (paragraph.payload.spec) onSpec?.(paragraph.payload.spec);
                }}>应用已验证选股条件</button
              >{/if}{/if}
        </article>{/each}
    </div>{/if}
  {#if events.some((event) => event.type === 'usage')}<p class="context-note">
      已报告用量：{events.findLast((event) => event.type === 'usage')?.payload.prompt_tokens ??
        '未知'} 输入 / {events.findLast((event) => event.type === 'usage')?.payload
        .completion_tokens ?? '未知'} 输出 tokens。
    </p>{/if}
  {#if failure}<StatusPanel title="AI 请求未完成" error={failure} />{/if}
</section>
