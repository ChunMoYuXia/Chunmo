<script lang="ts">
  import { ApiError } from '$lib/api/client';
  let {
    title = '',
    message = '',
    error = null,
    loading = false,
  }: { title?: string; message?: string; error?: unknown; loading?: boolean } = $props();
  let failure = $derived(error instanceof Error ? error.message : error ? String(error) : '');
  let apiError = $derived(error instanceof ApiError ? error : null);
</script>

<div
  class:error={Boolean(error)}
  class="status-panel"
  role={error ? 'alert' : 'status'}
  aria-live="polite"
>
  {#if title}<strong>{title}</strong>{/if}
  {#if loading}<span class="loading-label">处理中…</span>{/if}
  {#if message || failure}<p>{failure || message}</p>{/if}
  {#if apiError}
    <small
      >{apiError.code}
      {#if apiError.traceId}· 诊断编号 {apiError.traceId}{/if}</small
    >
    {#if Array.isArray(apiError.details.issues)}
      {#each apiError.details.issues as issue, index (index)}
        {#if issue && typeof issue === 'object' && 'message' in issue}<p>
            {String(issue.message)}
          </p>{/if}
      {/each}
    {/if}
  {/if}
</div>
