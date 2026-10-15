<script lang="ts">
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { Security } from '$lib/api/types';
  import { marketQuery, type ResearchContext } from '$lib/state/research';
  let {
    context,
    onSelect,
    id = 'security-search',
  }: { context: ResearchContext; onSelect: (security: Security) => void; id?: string } = $props();
  let query = $state('');
  let results = $state<Security[]>([]);
  let selected = $state(0);
  let opened = $state(false);
  let loading = $state(false);
  let failure = $state('');
  const scope = new RequestScope();
  $effect(() => {
    const active = context;
    const term = query.trim();
    const ticket = scope.start();
    results = [];
    selected = 0;
    failure = '';
    loading = false;
    if (!active.datasetId || !term) return () => scope.cancel();
    const timer = setTimeout(async () => {
      loading = true;
      try {
        const rows = await request<Security[]>(
          `/v1/market/search?${marketQuery(active, { q: term })}`,
          { signal: ticket.signal },
        );
        if (ticket.current()) {
          results = rows;
          opened = true;
        }
      } catch (error) {
        if (ticket.current()) failure = error instanceof Error ? error.message : '搜索失败';
      } finally {
        if (ticket.current()) loading = false;
      }
    }, 180);
    return () => {
      clearTimeout(timer);
      scope.cancel();
    };
  });
  function choose(security: Security) {
    opened = false;
    query = '';
    onSelect(security);
  }
  function keyboard(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      opened = false;
      return;
    }
    if (!results.length || !opened) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      selected = (selected + (event.key === 'ArrowDown' ? 1 : results.length - 1)) % results.length;
    } else if (event.key === 'Enter') {
      event.preventDefault();
      choose(results[selected]);
    }
  }
</script>

<div class="security-search">
  <label class="sr-only" for={id}>搜索股票</label>
  <input
    {id}
    type="search"
    role="combobox"
    aria-autocomplete="list"
    aria-expanded={opened && Boolean(results.length)}
    aria-controls={`${id}-results`}
    aria-activedescendant={opened && results[selected] ? `${id}-${selected}` : undefined}
    placeholder="代码、名称、拼音首字母或行业"
    disabled={!context.datasetId}
    bind:value={query}
    onfocus={() => (opened = true)}
    onkeydown={keyboard}
  />
  {#if loading}<small class="search-status">搜索中…</small>{/if}
  {#if failure}<small role="alert">{failure}</small>{/if}
  {#if opened && results.length}
    <div id={`${id}-results`} role="listbox" aria-label="股票搜索结果" class="search-results">
      {#each results as security, index (security.security_id)}
        <button
          id={`${id}-${index}`}
          type="button"
          role="option"
          aria-selected={index === selected}
          class:highlight={index === selected}
          onclick={() => choose(security)}
          ><strong>{security.code}</strong>
          {security.name}<small>{security.industry ?? '未分类'}</small></button
        >
      {/each}
    </div>
  {:else if opened && query.trim() && !loading && !failure}<small class="search-status"
      >暂无匹配股票</small
    >{/if}
</div>
