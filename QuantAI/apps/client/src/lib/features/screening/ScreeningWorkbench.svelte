<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { request } from '$lib/api/client';
  import { RequestScope } from '$lib/api/scope';
  import type {
    Feature,
    FieldDefinition,
    SavedScreening,
    ScreeningResult,
    ScreeningSpec,
  } from '$lib/api/types';
  import { marketQuery, researchContext, setResearchContext } from '$lib/state/research';
  import FeatureTable from '$lib/features/market/FeatureTable.svelte';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import UniversePicker from './UniversePicker.svelte';
  import FilterBuilder from './FilterBuilder.svelte';
  import Funnel from './Funnel.svelte';
  import Copilot from '$lib/features/ai/Copilot.svelte';
  import { draftsFromFilters, filtersFromDrafts, type ConditionDraft } from './drafts';

  let features = $state<Feature[]>([]);
  let fields = $state<FieldDefinition[]>([]);
  let selected = $state<string[]>([]);
  let drafts = $state<ConditionDraft[]>([]);
  let result = $state<ScreeningResult | null>(null);
  let presets = $state<SavedScreening[]>([]);
  let pendingPreset = $state<SavedScreening | null>(null);
  let name = $state('选股方案');
  let knowledgeMode = $state<NonNullable<ScreeningSpec['knowledge_mode']>>('public_pit');
  let sortBy = $state<FieldDefinition['key']>('momentum_20d');
  let descending = $state(true);
  let limit = $state(20);
  let loading = $state(false);
  let running = $state(false);
  let failure = $state<unknown>(null);
  let savedMessage = $state('');
  let showAI = $state(false);
  const scope = new RequestScope();
  const runScope = new RequestScope();
  let parsed = $derived.by(() => {
    try {
      return { filters: filtersFromDrafts(drafts, fields), error: '' };
    } catch (error) {
      return { filters: [], error: error instanceof Error ? error.message : '条件不合法' };
    }
  });

  $effect(() => {
    const context = $researchContext;
    const mode = knowledgeMode;
    const ticket = scope.start();
    runScope.cancel();
    running = false;
    features = [];
    selected = [];
    result = null;
    failure = null;
    loading = Boolean(context.datasetId);
    if (!context.datasetId) return () => scope.cancel();
    Promise.all([
      request<Feature[]>(`/v1/market/snapshot?${marketQuery(context, { knowledge_mode: mode })}`, {
        signal: ticket.signal,
      }),
      request<FieldDefinition[]>('/v1/screening/fields', { signal: ticket.signal }),
      request<SavedScreening[]>('/v1/screening/presets', { signal: ticket.signal }),
    ])
      .then(([rows, definitions, saved]) => {
        if (!ticket.current()) return;
        features = rows;
        fields = definitions;
        presets = saved;
        selected = rows.map((row) => row.security_id);
        const toApply = pendingPreset;
        pendingPreset = null;
        if (
          toApply &&
          toApply.spec.dataset_id === context.datasetId &&
          Date.parse(toApply.spec.as_of) === Date.parse(context.asOf)
        ) {
          selected = toApply.spec.universe;
          drafts = draftsFromFilters(toApply.spec.filters ?? [], definitions);
          sortBy = toApply.spec.sort_by ?? 'momentum_20d';
          descending = toApply.spec.descending ?? true;
          limit = toApply.spec.limit ?? 20;
          name = toApply.name;
          void run(toApply.spec);
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
      runScope.cancel();
    };
  });

  function specification(): ScreeningSpec {
    if (parsed.error) throw new Error(parsed.error);
    return {
      version: 1,
      dataset_id: $researchContext.datasetId,
      as_of: $researchContext.asOf,
      universe: selected,
      filters: parsed.filters,
      sort_by: sortBy,
      descending,
      limit,
      knowledge_mode: knowledgeMode,
    };
  }

  async function run(override?: ScreeningSpec) {
    const ticket = runScope.start();
    running = true;
    failure = null;
    result = null;
    try {
      const data = await request<ScreeningResult>('/v1/screening', {
        method: 'POST',
        body: JSON.stringify(override ?? specification()),
        signal: ticket.signal,
      });
      if (ticket.current()) result = data;
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) running = false;
    }
  }

  function rollback(prefix: number) {
    drafts = drafts.slice(0, prefix);
    const filters = filtersFromDrafts(drafts, fields);
    void run({ ...specification(), filters });
  }
  async function save(transfer = false) {
    failure = null;
    savedMessage = '';
    try {
      const saved = await request<SavedScreening>('/v1/screening/presets', {
        method: 'POST',
        body: JSON.stringify({ name, spec: specification() }),
      });
      presets = [saved, ...presets];
      savedMessage = `已保存方案：${saved.name}`;
      if (transfer) await goto(resolve(`/portfolios?screening=${encodeURIComponent(saved.id)}`));
    } catch (error) {
      failure = error;
    }
  }
</script>

<svelte:head><title>选股 · 组合研究台</title></svelte:head>
<label class="actions"
  >时间口径<select aria-label="选股时间口径" bind:value={knowledgeMode}
    ><option value="public_pit">公开历史</option><option value="system_replay"
      >当时已入库的数据</option
    ></select
  ></label
>
<div class="page-heading">
  <div>
    <h1>选股工作台</h1>
    <p>从候选池到可解释的筛选结果，保留每一步依据。</p>
  </div>
  <label
    >已保存方案<select
      aria-label="已保存方案"
      value=""
      onchange={(event) => {
        const preset = presets.find((item) => item.id === event.currentTarget.value);
        if (preset) {
          pendingPreset = preset;
          knowledgeMode = preset.spec.knowledge_mode ?? 'public_pit';
          setResearchContext({ datasetId: preset.spec.dataset_id, asOf: preset.spec.as_of });
        }
      }}
      ><option value="">选择方案重载</option>{#each presets as preset (preset.id)}<option
          value={preset.id}>{preset.name}</option
        >{/each}</select
    ></label
  >
</div>
{#if !$researchContext.datasetId}<StatusPanel
    title="先选择研究数据"
    message="在数据中心导入 CSV，或创建合成演示数据。"
  />
{:else if loading}<StatusPanel title="正在加载候选池" loading />
{:else}<div class="workbench">
    <div class="stack">
      <UniversePicker
        {features}
        {selected}
        onChange={(ids) => {
          selected = ids;
          result = null;
        }}
      />
      {#if result}<section class="panel" aria-label="筛选结果">
          <div class="section-heading">
            <h2>筛选结果</h2>
            <span class="badge">保留 {result.total_matching_count} 只</span>
          </div>
          <FeatureTable features={result.rows} />{#if !result.rows.length}<StatusPanel
              title="没有匹配的股票"
              message="调整筛选条件，或检查历史窗口是否完整。"
            />{/if}
          <p class="context-note">
            显示排序后的前 {result.rows.length} 只；空值按缺失原因剔除。{knowledgeMode ===
            'public_pit'
              ? '公开历史模式。'
              : '当时已入库的数据；后补数据不可见。'}
          </p>
        </section>{/if}
    </div>
    <aside class="stack">
      <details bind:open={showAI}>
        <summary>使用自然语言生成筛选条件</summary>{#if showAI}<Copilot
            scope={{
              version: 1,
              dataset_id: $researchContext.datasetId,
              as_of: $researchContext.asOf,
              universe: selected,
              filters: [],
              sort_by: sortBy,
              descending,
              limit,
              knowledge_mode: knowledgeMode,
            }}
            onSpec={(spec) => {
              if (
                spec.dataset_id !== $researchContext.datasetId ||
                Date.parse(spec.as_of) !== Date.parse($researchContext.asOf) ||
                (spec.knowledge_mode ?? 'public_pit') !== knowledgeMode
              )
                return;
              selected = spec.universe;
              drafts = draftsFromFilters(spec.filters ?? [], fields);
              sortBy = spec.sort_by ?? 'momentum_20d';
              descending = spec.descending ?? true;
              limit = spec.limit ?? 20;
              result = null;
            }}
          />
        {/if}
      </details>
      <section class="panel">
        <div class="section-heading">
          <h2>筛选条件</h2>
          <small>最多 20 条</small>
        </div>
        <FilterBuilder
          {drafts}
          {fields}
          onChange={(value) => {
            drafts = value;
            result = null;
          }}
        />
        <div class="form-grid" style="margin-top:18px">
          <label
            >排序指标<select bind:value={sortBy}
              >{#each fields as field (field.key)}<option value={field.key}>{field.label}</option
                >{/each}</select
            ></label
          ><label>显示数量<input type="number" min="1" max="500" bind:value={limit} /></label>
        </div>
        <label class="checkbox-label"
          ><input type="checkbox" bind:checked={descending} />从高到低排列</label
        >
        {#if parsed.error}<p role="alert" class="warning">{parsed.error}</p>{/if}
        <div class="actions" style="margin-top:18px">
          <button
            class="primary"
            disabled={running || loading || Boolean(parsed.error)}
            onclick={() => void run()}>{running ? '正在筛选…' : '运行筛选'}</button
          >
        </div>
        <hr />
        <label>方案名称<input bind:value={name} maxlength="80" /></label>
        <div class="actions" style="margin-top:12px">
          <button disabled={!name.trim() || Boolean(parsed.error)} onclick={() => void save()}
            >保存方案</button
          >
        </div>
        <button
          disabled={!result?.rows.length || !name.trim() || Boolean(parsed.error)}
          onclick={() => void save(true)}>保存并带入组合</button
        >
        {#if savedMessage}<p role="status" class="context-note">{savedMessage}</p>{/if}
      </section>
      {#if result}<section class="panel">
          <Funnel
            steps={result.steps}
            universeCount={result.universe_count}
            matchingCount={result.total_matching_count}
            onRollback={rollback}
          />
        </section>{/if}
    </aside>
  </div>{/if}
{#if failure}<StatusPanel title="选股操作失败" error={failure} />{/if}
