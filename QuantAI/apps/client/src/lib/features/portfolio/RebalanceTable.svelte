<script lang="ts">
  import type { components } from '$lib/api/schema';
  import { number } from '$lib/utils/format';
  let { plan }: { plan: components['schemas']['RebalanceDTO'] } = $props();
</script>

<section class="panel">
  <div class="section-heading">
    <h2>整股调仓预览</h2>
    <span class="badge">只生成计划</span>
  </div>
  <div class="stats-row">
    <div class="stat">
      <small>估计费用 / 元</small><strong>{number(plan.estimated_fees)}</strong>
    </div>
    <div class="stat">
      <small>调仓后现金 / 元</small><strong>{number(plan.cash_after)}</strong>
    </div>
    <div class="stat">
      <small>调仓前权益 / 元</small><strong>{number(plan.equity_before)}</strong>
    </div>
  </div>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>证券</th><th>方向</th><th>股数</th><th>参考价 / 元</th><th>佣金 / 元</th><th
            >过户费 / 元</th
          ><th>印花税 / 元</th></tr
        ></thead
      ><tbody
        >{#each plan.orders as order (order.security_id + order.side)}<tr
            ><td>{order.security_id}</td><td>{order.side === 'buy' ? '买入' : '卖出'}</td><td
              >{order.quantity}</td
            ><td>{number(order.reference_price)}</td><td>{number(order.fees.commission)}</td><td
              >{number(order.fees.transfer)}</td
            ><td>{number(order.fees.stamp)}</td></tr
          >{/each}</tbody
      >
    </table>
  </div>
  <p class="context-note">
    费率版本 {plan.fee_schedule.version}；实际佣金需按券商配置。调仓计划不会自动提交模拟订单。
  </p>
  {#each plan.assumptions as note (note)}<p class="context-note">{note}</p>{/each}
</section>
