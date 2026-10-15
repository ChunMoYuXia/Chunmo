<script lang="ts">
  import { onMount, type Snippet } from 'svelte';
  import { isTauri, invoke } from '@tauri-apps/api/core';
  import { connectBackend } from '$lib/api/bootstrap';
  import type { BackendConnection } from '$lib/api/client';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import AppShell from '$lib/components/AppShell.svelte';
  import '$lib/styles/tokens.css';

  let { children }: { children: Snippet } = $props();
  let ready = $state(false);
  let failure = $state<unknown>(null);

  async function initialize() {
    failure = null;
    try {
      await connectBackend(
        new URL(window.location.href),
        (url) => window.history.replaceState(null, '', url),
        isTauri() ? () => invoke<BackendConnection>('backend_connection') : undefined,
      );
      ready = true;
    } catch (error) {
      failure = error;
    }
  }

  onMount(() => {
    void initialize();
  });
</script>

{#if ready}
  <AppShell>{@render children()}</AppShell>
{:else}
  <main class="connection-state" aria-live="polite">
    <h1>组合研究台</h1>
    {#if failure}
      <StatusPanel title="无法连接本地研究服务" error={failure} />
      <p>请通过本地客户端重新打开工作台。</p>
      <button onclick={initialize}>重新连接</button>
    {:else}
      <p>正在连接本地研究服务…</p>
    {/if}
  </main>
{/if}

<style>
  :global(body) {
    margin: 0;
    background: #121b28;
    color: #e5edf7;
    font-family: 'Microsoft YaHei UI', sans-serif;
  }
  .connection-state {
    max-width: 34rem;
    margin: 15vh auto;
    padding: 2rem;
  }
  p {
    color: #a2b3ca;
    line-height: 1.8;
  }
  button {
    border: 1px solid #4d8fff;
    border-radius: 6px;
    padding: 0.65rem 1.2rem;
    background: #23487c;
    color: white;
    cursor: pointer;
  }
</style>
