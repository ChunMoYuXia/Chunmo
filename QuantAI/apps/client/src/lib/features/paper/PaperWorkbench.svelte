<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { request } from '$lib/api/client';
  import { CommandClient } from '$lib/api/commands';
  import { RequestScope } from '$lib/api/scope';
  import type { components } from '$lib/api/schema';
  import type { Security } from '$lib/api/types';
  import { researchContext, marketQuery } from '$lib/state/research';
  import { number } from '$lib/utils/format';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import AccountSelector from './AccountSelector.svelte';
  import HaltButton from './HaltButton.svelte';
  import OrderForm from './OrderForm.svelte';
  import LedgerTable from './LedgerTable.svelte';
  let orderBusy = $state(false);
  type Account = components['schemas']['AccountSnapshot'];
  type Order = components['schemas']['OrderDTO'];
  let accounts = $state<Account[]>([]),
    id = $state(''),
    account = $state<Account | null>(null),
    orders = $state<Order[]>([]),
    ledger = $state<components['schemas']['LedgerEvent'][]>([]),
    securities = $state<Security[]>([]);
  let name = $state('模拟账户'),
    capital = $state('100000'),
    start = $state(''),
    busy = $state(false),
    loading = $state(false),
    failure = $state<unknown>(null),
    audit = $state<components['schemas']['AuditVerification'] | null>(null);
  const scope = new RequestScope(),
    listScope = new RequestScope();
  const createCommand = new CommandClient('paper-create'),
    stepCommand = new CommandClient('paper-step'),
    cancelCommand = new CommandClient('paper-cancel');
  const states: Record<string, string> = {
    submitted: '已提交',
    filled: '已成交',
    partially_filled: '部分成交',
    canceled: '已撤销',
    expired: '已过期',
    rejected: '已拒绝',
  };
  async function list() {
    const ticket = listScope.start();
    try {
      const values = await request<Account[]>('/v1/paper/accounts', { signal: ticket.signal });
      if (ticket.current()) {
        accounts = values;
        if (!id) id = values[0]?.id ?? '';
      }
    } catch (error) {
      if (ticket.current()) failure = error;
    }
  }
  onMount(() => {
    void list();
    return () => {
      scope.cancel();
      listScope.cancel();
    };
  });
  $effect(() => {
    const selected = id;
    account = null;
    orders = [];
    ledger = [];
    securities = [];
    audit = null;
    failure = null;
    if (selected) untrack(() => void reload(selected));
    else scope.cancel();
  });
  async function reload(selected = id) {
    const ticket = scope.start();
    loading = true;
    failure = null;
    try {
      const base = `/v1/paper/accounts/${encodeURIComponent(selected)}`;
      const [value, pending, events] = await Promise.all([
        request<Account>(base, { signal: ticket.signal }),
        request<Order[]>(`${base}/orders`, { signal: ticket.signal }),
        request<components['schemas']['LedgerEvent'][]>(`${base}/ledger?limit=200`, {
          signal: ticket.signal,
        }),
      ]);
      if (!ticket.current()) return;
      account = value;
      orders = pending;
      ledger = events;
      audit = null;
      const query = marketQuery({
        datasetId: value.dataset_id,
        asOf: new Date(`${value.session}T15:00:00+08:00`).toISOString(),
      });
      const assets = await request<Security[]>(`/v1/market/securities?${query}`, {
        signal: ticket.signal,
      });
      if (ticket.current()) securities = assets;
    } catch (error) {
      if (ticket.current()) failure = error;
    } finally {
      if (ticket.current()) loading = false;
    }
  }
  async function create() {
    busy = true;
    failure = null;
    try {
      const value = await createCommand.send<Account>('/v1/paper/accounts', {
        name,
        dataset_id: $researchContext.datasetId,
        capital,
        start: start || null,
      });
      accounts = [value, ...accounts];
      id = value.id;
      await list();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
  async function advance() {
    if (!account) return;
    busy = true;
    failure = null;
    try {
      await stepCommand.send(`/v1/paper/accounts/${encodeURIComponent(account.id)}/advance`, {});
      await reload();
      await list();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
  async function cancel(order: Order) {
    busy = true;
    failure = null;
    try {
      await cancelCommand.send(
        `/v1/paper/accounts/${encodeURIComponent(order.account_id)}/orders/${encodeURIComponent(order.id)}/cancel`,
        {},
      );
      await reload();
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
  async function more() {
    if (!account) return;
    const selected = id;
    const after = ledger.at(-1)?.seq ?? 0;
    try {
      const events = await request<components['schemas']['LedgerEvent'][]>(
        `/v1/paper/accounts/${encodeURIComponent(id)}/ledger?after=${after}&limit=200`,
      );
      if (id === selected)
        ledger = [
          ...ledger,
          ...events.filter((event) => !ledger.some((existing) => existing.seq === event.seq)),
        ];
    } catch (error) {
      failure = error;
    }
  }
  async function verify() {
    const selected = id;
    try {
      const result = await request<components['schemas']['AuditVerification']>(
        `/v1/paper/accounts/${encodeURIComponent(id)}/audit/verify`,
      );
      if (selected === id) audit = result;
    } catch (error) {
      failure = error;
    }
  }
</script>

<svelte:head><title>模拟盘 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>本地模拟盘</h1>
    <p>手动推进交易日，核对每笔订单、费用和账本事件。</p>
  </div>
  <button onclick={() => void reload()} disabled={!id || loading}>刷新账户</button>
</div>
<div class="workbench">
  <div class="stack">
    <section class="panel">
      <AccountSelector
        disabled={busy || loading || orderBusy}
        {accounts}
        selectedId={id}
        onSelect={(value) => {
          id = value;
        }}
      />
      {#if account}<div class="section-heading">
          <p class="context-note">
            账户时间 {account.session} · v{account.data_version} · {account.source_kind ===
            'synthetic'
              ? '合成演示'
              : '本地导入'} · {account.status === 'paused' ? '已暂停' : '可提交订单'}
          </p>
          <HaltButton
            accountId={account.id}
            paused={account.status === 'paused'}
            onChange={() => void reload()}
          />
        </div>
        <div class="stats-row">
          <div class="stat">
            <small>权益 / 元</small><strong>{number(account.nav)}</strong>
          </div>
          <div class="stat">
            <small>可用现金 / 元</small><strong>{number(account.available_cash)}</strong>
          </div>
          <div class="stat">
            <small>冻结现金 / 元</small><strong>{number(account.frozen_cash)}</strong>
          </div>
          <div class="stat">
            <small>应收款 / 元</small><strong>{number(account.receivables)}</strong>
          </div>
        </div>
        <p class="context-note">
          现金余额 {number(account.cash)} 元 · 可提现 {number(account.withdrawable_cash)} 元。研究顶部日期不改变模拟账户时间。
        </p>
        <button
          class="primary"
          disabled={busy || loading || orderBusy || account.status === 'paused'}
          onclick={() => void advance()}>推进下一交易日</button
        >
        {#if account.stale_securities.length}<p class="warning">
            估值沿用旧价格：{account.stale_securities.join('、')}
          </p>{/if}
        <details>
          <summary>账户撮合假设</summary>{#each account.assumptions as note (note)}<p
              class="context-note"
            >
              {note}
            </p>{/each}
        </details>{/if}
    </section>
    {#if loading}<StatusPanel title="正在重载持久化账户" loading />{/if}
    {#if account}<section class="panel">
        <h2>持仓与可卖数量</h2>
        <div class="table-scroll">
          <table>
            <thead
              ><tr
                ><th>证券</th><th>持仓 / 股</th><th>可卖 / 股</th><th>卖单占用 / 股</th><th
                  >成本金额 / 元</th
                ></tr
              ></thead
            ><tbody
              >{#each Object.entries(account.positions) as [security, position] (security)}<tr
                  ><td>{security}</td><td>{position.quantity}</td><td
                    >{account.available_to_sell[security] ?? 0}</td
                  ><td>{account.reserved_shares[security] ?? 0}</td><td
                    >{number(position.cost_basis)}</td
                  ></tr
                >{/each}</tbody
            >
          </table>
        </div>
      </section>
      <section class="panel">
        <h2>订单记录</h2>
        <div class="table-scroll">
          <table>
            <thead
              ><tr
                ><th>证券</th><th>方向</th><th>委托 / 股</th><th>成交 / 股</th><th>最早执行</th><th
                  >状态 / 原因</th
                ><th>操作</th></tr
              ></thead
            ><tbody
              >{#each orders as order (order.id)}<tr
                  ><td>{order.security_id}<small>{order.id.slice(0, 12)}</small></td><td
                    >{order.side === 'buy' ? '买入' : '卖出'}</td
                  ><td>{order.quantity}</td><td>{order.filled_quantity}</td><td
                    >{order.eligible_session}</td
                  ><td
                    >{states[order.state] ?? `未知状态 ${order.state}`}<small
                      >{order.reason ?? ''}</small
                    ></td
                  ><td
                    >{#if order.state === 'submitted' || order.state === 'partially_filled'}<button
                        disabled={busy}
                        onclick={() => void cancel(order)}>撤单</button
                      >{/if}</td
                  ></tr
                >{/each}</tbody
            >
          </table>
        </div>
      </section>
      <section class="panel">
        <div class="section-heading">
          <h2>资金与股份账本</h2>
          <small>{ledger.length} / {account.ledger_count} 条</small>
        </div>
        <LedgerTable events={ledger} />{#if ledger.length < account.ledger_count}<button
            onclick={() => void more()}>加载更多账本</button
          >{/if}
        <div class="actions">
          <button onclick={() => void verify()}>核验审计链</button>{#if audit}<span
              class:warning={!audit.valid}
              role="status"
              >{audit.valid ? '核验通过' : '核验失败'} · {audit.checked_events} 个事件 · {audit.reason ??
                ''}</span
            >{/if}
        </div>
        <p class="context-note">
          核验覆盖审计链及账户、订单、成交、账本投影；无法防止管理员重签整条链。
        </p>
      </section>{/if}
  </div>
  <aside class="stack">
    {#if account}<section class="panel">
        <h2>模拟委托</h2>
        <OrderForm
          disabled={busy || loading}
          onBusyChange={(value) => (orderBusy = value)}
          accountId={account.id}
          paused={account.status === 'paused'}
          {securities}
          onChange={() => void reload()}
        />
      </section>{/if}
    <section class="panel">
      <h2>新建模拟账户</h2>
      <p class="context-note">使用顶部选定数据版本；留空开始日期时保留 20 日历史预热。</p>
      <form
        onsubmit={(event) => {
          event.preventDefault();
          void create();
        }}
      >
        <fieldset disabled={busy || orderBusy}>
          <label>账户名称<input bind:value={name} maxlength="80" required /></label><label
            >账户初始资金 / 元<input bind:value={capital} inputmode="decimal" required /></label
          ><label>起始交易日（可选）<input type="date" bind:value={start} /></label><button
            disabled={!$researchContext.datasetId}
            class="primary">创建模拟账户</button
          >
        </fieldset>
      </form>
    </section>
  </aside>
</div>
{#if failure}<StatusPanel title="模拟操作失败" error={failure} />{/if}
