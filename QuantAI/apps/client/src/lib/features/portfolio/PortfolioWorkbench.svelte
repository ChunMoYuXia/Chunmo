<script lang="ts">
  import { untrack } from 'svelte';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import type { Feature, SavedScreening, ScreeningResult } from '$lib/api/types';
  import { marketQuery, researchContext, setResearchContext } from '$lib/state/research';
  import { number, percentInputToRatio, ratioInput } from '$lib/utils/format';
  import UniversePicker from '$lib/features/screening/UniversePicker.svelte';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import WeightEditor from './WeightEditor.svelte';
  import OptimizationForm from './OptimizationForm.svelte';
  import RebalanceTable from './RebalanceTable.svelte';
  type Portfolio = components['schemas']['PortfolioDTO'];
  let rows = $state<Feature[]>([]),
    saved = $state<Portfolio[]>([]),
    weights = $state<Record<string, string>>({}),
    holdings = $state<Record<string, number>>({});
  let name = $state('研究组合'),
    cash = $state('100000'),
    cap = $state('100'),
    mode = $state<'public_pit' | 'system_replay'>('public_pit');
  let active = $state<Portfolio | null>(null),
    plan = $state<components['schemas']['RebalanceDTO'] | null>(null),
    solution = $state<components['schemas']['OptimizationDTO'] | null>(null);
  let loading = $state(false),
    busy = $state(false),
    failure = $state<unknown>(null),
    message = $state('');
  let weightsValid = $state(true);
  let loadedScreening = '';
  let draftContext = '';
  const scope = new RequestScope();
  const actions = new RequestScope();
  let selected = $derived(Object.keys(weights));
  $effect(() => {
    const context = $researchContext;
    const screening = page.url.searchParams.get('screening');
    const ticket = scope.start();
    actions.cancel();
    busy = false;
    plan = null;
    solution = null;
    failure = null;
    rows = [];
    if (!context.datasetId) return () => scope.cancel();
    loading = true;
    untrack(() => {
      if (draftContext !== `${context.datasetId}:${context.asOf}`) {
        draftContext = `${context.datasetId}:${context.asOf}`;
        if (
          !active ||
          active.spec.dataset_id !== context.datasetId ||
          Date.parse(active.spec.as_of) !== Date.parse(context.asOf)
        ) {
          active = null;
          weights = {};
          holdings = {};
        }
      }
    });
    Promise.all([
      request<Feature[]>(`/v1/market/snapshot?${marketQuery(context)}`, { signal: ticket.signal }),
      request<Portfolio[]>('/v1/portfolios', { signal: ticket.signal }),
    ])
      .then(async ([features, portfolios]) => {
        if (!ticket.current()) return;
        rows = features;
        saved = portfolios;
        if (screening && screening !== loadedScreening) {
          const preset = await request<SavedScreening>(
            `/v1/screening/presets/${encodeURIComponent(screening)}`,
            { signal: ticket.signal },
          );
          if (!ticket.current()) return;
          if (
            preset.spec.dataset_id !== context.datasetId ||
            Date.parse(preset.spec.as_of) !== Date.parse(context.asOf)
          ) {
            setResearchContext({ datasetId: preset.spec.dataset_id, asOf: preset.spec.as_of });
            return;
          }
          const result = await request<ScreeningResult>('/v1/screening', {
            method: 'POST',
            signal: ticket.signal,
            body: JSON.stringify(preset.spec),
          });
          if (!ticket.current()) return;
          weights = Object.fromEntries(result.rows.map((row) => [row.security_id, '0']));
          holdings = {};
          name = `${preset.name}组合`;
          mode = preset.spec.knowledge_mode ?? 'public_pit';
          active = null;
          loadedScreening = screening;
          message = `已带入 ${result.rows.length} 只筛选证券，请配置目标权重。`;
        }
      })
      .catch((error) => {
        if (ticket.current()) failure = error;
      })
      .finally(() => {
        if (ticket.current()) loading = false;
      });
    return () => {
      scope.cancel();
      actions.cancel();
    };
  });
  function invalidateDraft() {
    actions.cancel();
    busy = false;
    active = null;
    plan = null;
    solution = null;
    message = '';
  }
  function edit(value: Record<string, string>) {
    invalidateDraft();
    weights = value;
  }
  function load(value: Portfolio) {
    actions.cancel();
    busy = false;
    failure = null;
    name = value.spec.name;
    weights = value.spec.target_weights;
    holdings = value.spec.holdings ?? {};
    cash = value.spec.cash ?? '100000';
    cap = ratioInput(value.spec.max_weight ?? '1');
    mode = value.spec.knowledge_mode ?? 'public_pit';
    active = value;
    setResearchContext({ datasetId: value.spec.dataset_id, asOf: value.spec.as_of });
    plan = null;
    solution = null;
    message = '已重载保存的组合版本。';
  }
  async function save() {
    const ticket = actions.start();
    busy = true;
    failure = null;
    try {
      const result = await request<Portfolio>('/v1/portfolios', {
        method: 'POST',
        signal: ticket.signal,
        body: JSON.stringify({
          name,
          dataset_id: $researchContext.datasetId,
          as_of: $researchContext.asOf,
          target_weights: weights,
          max_weight: percentInputToRatio(cap),
          holdings,
          cash,
          knowledge_mode: mode,
        }),
      });
      if (ticket.current()) {
        active = result;
        saved = [result, ...saved];
        message = '组合已保存。';
      }
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) busy = false;
    }
  }
  async function rebalance() {
    if (!active) return;
    const ticket = actions.start();
    busy = true;
    failure = null;
    try {
      const result = await request<components['schemas']['RebalanceDTO']>(
        `/v1/portfolios/${encodeURIComponent(active.id)}/rebalance`,
        { method: 'POST', body: '{}', signal: ticket.signal },
      );
      if (ticket.current()) plan = result;
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) busy = false;
    }
  }
</script>

<svelte:head><title>组合 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>组合配置</h1>
    <p>配置目标权重、现有持仓与现金，保存后可用于回测与风险分析。</p>
  </div>
  <label
    >已保存组合<select
      aria-label="已保存组合"
      value=""
      onchange={(event) => {
        const item = saved.find((p) => p.id === event.currentTarget.value);
        if (item) load(item);
      }}
      ><option value="">选择组合重载</option>{#each saved as item (item.id)}<option value={item.id}
          >{item.spec.name} · {item.created_at.slice(0, 10)}</option
        >{/each}</select
    ></label
  >
</div>
{#if !$researchContext.datasetId}<StatusPanel
    title="先在数据中心准备研究数据"
  />{:else if loading}<StatusPanel title="正在加载组合环境" loading />{:else}
  <div class="workbench">
    <div class="stack">
      <UniversePicker
        features={rows}
        {selected}
        onChange={(ids) => edit(Object.fromEntries(ids.map((id) => [id, weights[id] ?? '0'])))}
      />
      <section class="panel">
        <h2>目标权重与持仓</h2>
        <WeightEditor
          {weights}
          onChange={edit}
          onValidityChange={(valid) => {
            weightsValid = valid;
            if (!valid) invalidateDraft();
          }}
        />
        <details>
          <summary>录入当前持仓 / 股</summary>
          <p class="context-note">
            此处为保存的研究持仓。可卖数量在调仓中默认等于持仓，模拟账户独立维护 T+1。
          </p>
          <div class="form-grid">
            {#each selected as id (id)}<label
                >{id}<input
                  type="number"
                  min="0"
                  step="1"
                  value={holdings[id] ?? 0}
                  oninput={(event) => {
                    holdings = { ...holdings, [id]: Number(event.currentTarget.value) };
                    invalidateDraft();
                  }}
                /></label
              >{/each}
          </div>
        </details>
      </section>
      {#if plan}<RebalanceTable {plan} />{/if}
    </div>
    <aside class="stack">
      <section class="panel">
        <h2>配置约束</h2>
        <label>组合名称<input bind:value={name} maxlength="80" oninput={invalidateDraft} /></label
        ><label
          >现金 / 元<input bind:value={cash} inputmode="decimal" oninput={invalidateDraft} /></label
        ><label
          >个股上限 / %<input
            bind:value={cap}
            inputmode="decimal"
            oninput={invalidateDraft}
          /></label
        ><label
          >时间口径<select bind:value={mode} onchange={invalidateDraft}
            ><option value="public_pit">公开历史</option><option value="system_replay"
              >当时已入库</option
            ></select
          ></label
        >
        <OptimizationForm
          context={$researchContext}
          universe={selected}
          maxWeight={/^(\d+)(\.\d*)?$/.test(cap) ? percentInputToRatio(cap) : '0'}
          knowledgeMode={mode}
          onResult={(value) => {
            edit(value.target_weights);
            solution = value;
          }}
        />
        {#if solution}<p class="context-note">
            {solution.observations} 个历史观测；价格口径 {solution.price_basis}。
          </p>
          {#each solution.warnings as warning (warning)}<p class="warning">{warning}</p>{/each}{/if}
        <div class="actions">
          <button
            class="primary"
            disabled={busy || !selected.length || !name.trim() || !weightsValid}
            onclick={() => void save()}>保存组合</button
          ><button disabled={busy || !active} onclick={() => void rebalance()}>预览调仓</button>
        </div>
        {#if active}<p class="context-note">
            保存版本现金权重 {number(Number(active.cash_weight) * 100)}%。
          </p>
          <div class="actions">
            <a href={resolve(`/backtests?portfolio=${encodeURIComponent(active.id)}`)}
              >回测此组合 →</a
            ><a href={resolve(`/risk?portfolio=${encodeURIComponent(active.id)}`)}>分析风险 →</a>
          </div>{/if}
        {#if message}<p role="status">{message}</p>{/if}
      </section>
    </aside>
  </div>{/if}
{#if failure}<StatusPanel title="组合操作失败" error={failure} />{/if}
