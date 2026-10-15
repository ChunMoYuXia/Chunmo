<script lang="ts">
  import type { components } from '$lib/api/schema';
  let {
    disabled = false,
    accounts,
    selectedId,
    onSelect,
  }: {
    disabled?: boolean;
    accounts: components['schemas']['AccountSnapshot'][];
    selectedId: string;
    onSelect: (id: string) => void;
  } = $props();
</script>

<label
  >模拟账户<select
    aria-label="模拟账户"
    {disabled}
    value={selectedId}
    onchange={(event) => onSelect(event.currentTarget.value)}
    ><option value="">选择账户</option>{#each accounts as account (account.id)}<option
        value={account.id}
        >{account.name} · {account.session} · {account.source_kind === 'synthetic'
          ? '合成演示'
          : '本地导入'}</option
      >{/each}</select
  ></label
>
