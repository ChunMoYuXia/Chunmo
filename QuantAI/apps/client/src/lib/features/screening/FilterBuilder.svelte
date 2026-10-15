<script lang="ts">
  import type { FieldDefinition, ScreeningFilter } from '$lib/api/types';
  import { newCondition, type ConditionDraft } from './drafts';
  let {
    drafts,
    fields,
    onChange,
  }: {
    drafts: ConditionDraft[];
    fields: FieldDefinition[];
    onChange: (drafts: ConditionDraft[]) => void;
  } = $props();
  const numericOperators = [
    { value: 'gte', label: '≥' },
    { value: 'lte', label: '≤' },
    { value: 'gt', label: '>' },
    { value: 'lt', label: '<' },
    { value: 'eq', label: '=' },
  ];
  function update(index: number, value: Partial<ConditionDraft>) {
    onChange(drafts.map((draft, current) => (current === index ? { ...draft, ...value } : draft)));
  }
</script>

<div aria-label="筛选条件">
  {#each drafts as draft, index (draft.key)}{@const definition = fields.find(
      (field) => field.key === draft.field,
    )}
    <div class="filter-row">
      <label
        ><span class="condition-label">指标 {index + 1}</span><select
          aria-label={`指标 ${index + 1}`}
          value={draft.field}
          onchange={(event) => {
            const field = fields.find((item) => item.key === event.currentTarget.value);
            if (field) update(index, { ...newCondition(field), key: draft.key });
          }}
          >{#each fields as field (field.key)}<option value={field.key}>{field.label}</option
            >{/each}</select
        ></label
      >
      <label
        ><span class="sr-only">比较方式 {index + 1}</span><select
          aria-label={`比较方式 ${index + 1}`}
          value={draft.operator}
          onchange={(event) =>
            update(index, { operator: event.currentTarget.value as ScreeningFilter['operator'] })}
        >
          {#if definition?.data_type === 'number'}{#each numericOperators as operator (operator.value)}<option
                value={operator.value}>{operator.label}</option
              >{/each}{:else}<option value="eq">=</option
            >{#if definition?.data_type === 'text'}<option value="in">属于</option>{/if}{/if}
        </select></label
      >
      <label
        ><span class="condition-label">条件值 {index + 1} {definition?.display_unit ?? ''}</span
        >{#if definition?.data_type === 'boolean'}<select
            aria-label={`条件值 ${index + 1}`}
            value={draft.displayValue}
            onchange={(event) => update(index, { displayValue: event.currentTarget.value })}
            ><option value="false">否</option><option value="true">是</option></select
          >{:else}<input
            aria-label={`条件值 ${index + 1}`}
            value={draft.displayValue}
            placeholder={draft.operator === 'in' ? '以逗号分隔' : ''}
            oninput={(event) => update(index, { displayValue: event.currentTarget.value })}
          />{/if}</label
      >
      <button
        aria-label={`删除条件 ${index + 1}`}
        onclick={() => onChange(drafts.filter((_, current) => current !== index))}>×</button
      >
    </div>{/each}
  <button
    disabled={!fields.length || drafts.length >= 20}
    onclick={() => {
      const field = fields.find((item) => item.key === 'momentum_20d') ?? fields[0];
      if (field) onChange([...drafts, newCondition(field)]);
    }}>添加条件</button
  >
</div>
