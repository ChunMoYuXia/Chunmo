<script lang="ts">
  import type { Feature } from '$lib/api/types';
  import FeatureTable from '$lib/features/market/FeatureTable.svelte';
  let {
    features,
    selected,
    onChange,
  }: { features: Feature[]; selected: string[]; onChange: (ids: string[]) => void } = $props();
</script>

<section class="panel" aria-label="候选池">
  <div class="section-heading">
    <h2>候选池 <small>{selected.length} / {features.length} 只</small></h2>
    <div class="actions">
      <button onclick={() => onChange(features.map((feature) => feature.security_id))}>全选</button
      ><button onclick={() => onChange([])}>清空</button>
    </div>
  </div>
  <FeatureTable
    {features}
    selectedIds={selected}
    onToggle={(id) =>
      onChange(selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id])}
  />
</section>
