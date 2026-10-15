<script lang="ts">
  import { CommandClient } from '$lib/api/commands';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  type SecurityOption = { security_id: string; name: string };
  let {
    accountId,
    paused,
    disabled = false,
    onBusyChange = () => {},
    securities,
    onChange,
  }: {
    accountId: string;
    paused: boolean;
    disabled?: boolean;
    onBusyChange?: (busy: boolean) => void;
    securities: SecurityOption[];
    onChange: () => void;
  } = $props();
  let securityId = $state(''),
    side = $state<'buy' | 'sell'>('buy'),
    quantity = $state(100),
    orderType = $state<'market' | 'limit'>('market'),
    price = $state('10');
  let busy = $state(false),
    failure = $state<unknown>(null),
    submitted = $state<components['schemas']['OrderDTO'] | null>(null);
  const command = new CommandClient('paper-order');
  $effect(() => {
    void accountId;
    securityId = securities[0]?.security_id ?? '';
    failure = null;
    submitted = null;
  });
  async function submit() {
    busy = true;
    onBusyChange(true);
    failure = null;
    submitted = null;
    try {
      submitted = await command.send<components['schemas']['OrderDTO']>(
        `/v1/paper/accounts/${encodeURIComponent(accountId)}/orders`,
        {
          security_id: securityId,
          side,
          quantity,
          order_type: orderType,
          limit_price: orderType === 'limit' ? price : null,
        },
      );
      onChange();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
      onBusyChange(false);
    }
  }
</script>

<form
  onsubmit={(event) => {
    event.preventDefault();
    void submit();
  }}
>
  <fieldset disabled={busy || paused || disabled}>
    <div class="form-grid">
      <label
        >委托证券<select bind:value={securityId} required
          >{#each securities as security (security.security_id)}<option value={security.security_id}
              >{security.name} · {security.security_id}</option
            >{/each}</select
        ></label
      >
      <label
        >方向<select bind:value={side}
          ><option value="buy">买入</option><option value="sell">卖出</option></select
        ></label
      >
      <label
        >委托数量 / 股<input
          type="number"
          min="1"
          max="2147483647"
          step="1"
          bind:value={quantity}
          required
        /></label
      >
      <label
        >订单类型<select bind:value={orderType}
          ><option value="market">市价</option><option value="limit">限价</option></select
        ></label
      >
      {#if orderType === 'limit'}<label
          >限价 / 元<input inputmode="decimal" bind:value={price} required /></label
        >{/if}
    </div>
    <button class="primary" disabled={!securityId}
      >{busy ? '正在提交…' : failure ? '重试同一订单' : '提交模拟订单'}</button
    >
  </fieldset>
</form>
<p class="context-note">
  仅模拟交易；收盘提交、下一交易日开盘撮合。超时重试沿用原幂等键；修改内容会产生新订单。请先核对持久化订单记录。
</p>
{#if paused}<p class="warning">账户已暂停，禁止新订单。</p>{/if}
{#if submitted}<StatusPanel
    title="模拟订单已记录"
    message={`${submitted.id} · ${submitted.state} · 最早执行 ${submitted.eligible_session}`}
  />{/if}
{#if failure}<StatusPanel title="订单未确认" error={failure} />{/if}
