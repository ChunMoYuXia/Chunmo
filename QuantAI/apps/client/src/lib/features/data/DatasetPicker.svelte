<script lang="ts">
  import type { Dataset } from '$lib/api/types';
  let {
    datasets,
    selectedId,
    onSelect,
  }: { datasets: Dataset[]; selectedId: string; onSelect: (dataset: Dataset) => void } = $props();
</script>

<label class="dataset-picker"
  ><span class="sr-only">研究数据集</span><select
    aria-label="研究数据集"
    value={selectedId}
    disabled={!datasets.length}
    onchange={(event) => {
      const selected = datasets.find((dataset) => dataset.dataset_id === event.currentTarget.value);
      if (selected) onSelect(selected);
    }}
  >
    {#if !datasets.length}<option value="">尚无数据集</option>{/if}
    {#each datasets as dataset (dataset.dataset_id)}<option value={dataset.dataset_id}
        >{dataset.name} · v{dataset.version}</option
      >{/each}
  </select></label
>
