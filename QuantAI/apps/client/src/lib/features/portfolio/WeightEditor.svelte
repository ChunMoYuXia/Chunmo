<script lang="ts">
  import { untrack } from 'svelte';
  import { percentInputToRatio, ratioInput } from '$lib/utils/format';
  let {
    weights,
    onChange,
    onValidityChange = () => {},
  }: {
    weights: Record<string, string>;
    onChange: (value: Record<string, string>) => void;
    onValidityChange?: (valid: boolean) => void;
  } = $props();
  let error = $state('');
  let inputs = $state<Record<string, string>>({});
  let previous: Record<string, string> = {};
  function validate() {
    try {
      Object.values(inputs).forEach(percentInputToRatio);
      error = '';
    } catch {
      error = '权重必须为有效小数';
    }
    onValidityChange(!error);
  }
  $effect(() => {
    const next = weights;
    untrack(() => {
      inputs = Object.fromEntries(
        Object.entries(next).map(([id, weight]) => {
          const text = inputs[id];
          if (text !== undefined) {
            if (previous[id] === weight) return [id, text];
            try {
              if (percentInputToRatio(text) === weight) return [id, text];
            } catch {
              /* external values replace invalid drafts */
            }
          }
          return [id, ratioInput(weight)];
        }),
      );
      previous = { ...next };
      validate();
    });
  });
  function edit(id: string, value: string) {
    inputs = { ...inputs, [id]: value };
    validate();
    try {
      const ratio = percentInputToRatio(value);
      onChange({ ...weights, [id]: ratio });
    } catch {
      error = '权重必须为有效小数';
    }
  }
</script>

<div class="table-scroll">
  <table>
    <thead><tr><th>证券</th><th>目标权重 / %</th><th>操作</th></tr></thead><tbody>
      {#each Object.entries(weights) as [id, weight] (id)}<tr
          ><td>{id}</td><td
            ><input
              aria-label={`${id} 目标权重 / %`}
              inputmode="decimal"
              value={inputs[id] ?? ratioInput(weight)}
              oninput={(event) => edit(id, event.currentTarget.value)}
            /></td
          ><td
            ><button
              onclick={() =>
                onChange(Object.fromEntries(Object.entries(weights).filter(([key]) => key !== id)))}
              >移除</button
            ></td
          ></tr
        >{/each}
    </tbody>
  </table>
</div>
{#if error}<p role="alert" class="warning">{error}</p>{/if}
<p class="context-note">输入百分比；剩余权重为现金。保存时由后端校验总和与个股上限。</p>
