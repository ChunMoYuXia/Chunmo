<script lang="ts">
  import type { components } from '$lib/api/schema';
  import { number } from '$lib/utils/format';
  let { events }: { events: components['schemas']['LedgerEvent'][] } = $props();
</script>

<div class="table-scroll">
  <table>
    <thead
      ><tr
        ><th>序号</th><th>日期</th><th>事项</th><th>证券</th><th>现金变动 / 元</th><th
          >冻结变动 / 元</th
        ><th>应收变动 / 元</th><th>股数变动</th><th>说明</th></tr
      ></thead
    ><tbody
      >{#each events as event (event.seq)}<tr
          ><td>{event.seq}</td><td>{event.session ?? '—'}</td><td>{event.kind}</td><td
            >{event.security_id ?? '—'}</td
          ><td>{number(event.delta_cash)}</td><td>{number(event.delta_frozen)}</td><td
            >{number(event.delta_receivables)}</td
          ><td>{event.delta_quantity ?? 0}</td><td>{event.description}</td></tr
        >{/each}</tbody
    >
  </table>
</div>
