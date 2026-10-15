<script lang="ts">
  import { CommandClient } from '$lib/api/commands';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let {
    accountId,
    paused,
    onChange,
  }: { accountId: string; paused: boolean; onChange: () => void } = $props();
  let isPaused = $state(false),
    busy = $state(false),
    confirming = $state(false),
    phrase = $state(''),
    message = $state('');
  let unknown = $state<string[]>([]),
    failure = $state<unknown>(null);
  const command = new CommandClient('paper-halt');
  $effect(() => {
    isPaused = paused;
    void accountId;
    confirming = false;
    message = '';
    unknown = [];
  });
  async function execute(resume: boolean) {
    busy = true;
    failure = null;
    try {
      const base = `/v1/paper/accounts/${encodeURIComponent(accountId)}`;
      if (resume) {
        await command.send(`${base}/resume`, { confirmation: phrase });
        isPaused = false;
        message = '已恢复模拟交易';
        confirming = false;
        phrase = '';
      } else {
        const result = await command.send<components['schemas']['HaltResult']>(`${base}/halt`, {});
        isPaused = result.account.status === 'paused';
        unknown = result.unknown_order_ids ?? [];
        message = '已暂停新订单';
      }
      onChange();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<div class="actions">
  {#if isPaused}<button disabled={busy} onclick={() => (confirming = true)}>恢复模拟交易</button
    >{:else}<button class="danger" disabled={busy} onclick={() => void execute(false)}
      >紧急停止</button
    >{/if}
  {#if message}<span role="status">{message}</span>{/if}
</div>
{#if unknown.length}<StatusPanel title="仍有未知状态订单" message={unknown.join('、')} />{/if}
{#if confirming}<div role="dialog" aria-label="恢复模拟交易确认" class="panel">
    <p>输入“恢复模拟交易”以允许新订单。</p>
    <label>恢复确认短语<input bind:value={phrase} autocomplete="off" /></label>
    <div class="actions">
      <button disabled={phrase !== '恢复模拟交易' || busy} onclick={() => void execute(true)}
        >确认恢复</button
      ><button onclick={() => (confirming = false)}>返回</button>
    </div>
  </div>{/if}
{#if failure}<StatusPanel title="账户操作失败" error={failure} />{/if}
