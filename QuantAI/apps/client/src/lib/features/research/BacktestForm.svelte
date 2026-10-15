<script lang="ts">
  import { CommandClient } from '$lib/api/commands';
  import type { components } from '$lib/api/schema';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  import { percentInputToRatio } from '$lib/utils/format';
  let {
    portfolio,
    dataset,
    onSubmit,
  }: {
    portfolio: components['schemas']['PortfolioDTO'];
    dataset: Pick<
      components['schemas']['DatasetSnapshot'],
      'dataset_id' | 'calendar_id' | 'first_session' | 'last_session'
    >;
    onSubmit: (id: string) => void;
  } = $props();
  let start = $state(''),
    end = $state(''),
    capital = $state('100000'),
    frequency = $state<'monthly' | 'weekly' | 'daily' | 'yearly'>('monthly'),
    slippage = $state('8'),
    participation = $state('5'),
    requireLimits = $state(true),
    benchmark = $state(''),
    annualization = $state(244),
    riskFree = $state('0');
  let busy = $state(false),
    failure = $state<unknown>(null);
  const command = new CommandClient('backtest-submit');
  let initializedContext = '';
  $effect(() => {
    const context = JSON.stringify([
      portfolio.id,
      dataset.dataset_id,
      dataset.calendar_id,
      dataset.first_session,
      dataset.last_session,
    ]);
    if (context === initializedContext) return;
    initializedContext = context;
    start = dataset.first_session;
    end = dataset.last_session;
    failure = null;
  });
  async function submit() {
    busy = true;
    failure = null;
    try {
      const payload: components['schemas']['BacktestRequest-Input'] = {
        dataset_id: dataset.dataset_id,
        calendar_id: dataset.calendar_id,
        security_ids: Object.keys(portfolio.spec.target_weights),
        weights: portfolio.spec.target_weights,
        start,
        end,
        capital,
        frequency,
        slippage_bps: slippage,
        participation_rate: percentInputToRatio(participation),
        require_price_limits: requireLimits,
        knowledge_mode: portfolio.spec.knowledge_mode ?? 'public_pit',
        benchmark_id: benchmark || null,
        annualization,
        risk_free_annual: Number(percentInputToRatio(riskFree)),
        seed: 42,
        fee_version: 'cn-stock-default',
      };
      const result = await command.send<components['schemas']['JobSubmission']>(
        '/v1/backtests',
        payload,
      );
      onSubmit(result.task_id);
    } catch (error) {
      failure = error;
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel">
  <h2>回测参数</h2>
  <p class="context-note">
    {portfolio.spec.name} · 使用已保存目标权重；固定配置策略，不在历史中重新选股。
  </p>
  <form
    onsubmit={(event) => {
      event.preventDefault();
      void submit();
    }}
  >
    <fieldset disabled={busy}>
      <div class="form-grid">
        <label
          >开始日期<input
            type="date"
            bind:value={start}
            min={dataset.first_session}
            max={dataset.last_session}
            required
          /></label
        ><label
          >结束日期<input
            type="date"
            bind:value={end}
            min={start}
            max={dataset.last_session}
            required
          /></label
        >
        <label>初始资金 / 元<input inputmode="decimal" bind:value={capital} required /></label
        ><label
          >调仓频率<select bind:value={frequency}
            ><option value="monthly">每月</option><option value="weekly">每周</option><option
              value="daily">每日</option
            ><option value="yearly">每年</option></select
          ></label
        >
        <label>滑点 / bp<input inputmode="decimal" bind:value={slippage} required /></label><label
          >历史成交量参与率 / %<input
            bind:value={participation}
            inputmode="decimal"
            required
          /></label
        >
        <label
          >基准证券<select bind:value={benchmark}
            ><option value="">不设基准</option
            >{#each Object.keys(portfolio.spec.target_weights) as id (id)}<option value={id}
                >{id} · 原始价格收益</option
              >{/each}</select
          ></label
        >
        <label
          >年化交易日<input
            type="number"
            min="1"
            max="1000"
            bind:value={annualization}
            required
          /></label
        ><label>年无风险利率 / %<input bind:value={riskFree} inputmode="decimal" required /></label>
      </div>
      <label class="checkbox-label"
        ><input type="checkbox" bind:checked={requireLimits} />要求价格涨跌停数据</label
      ><button class="primary">{busy ? '正在提交…' : '提交回测'}</button>
    </fieldset>
  </form>
  <p class="context-note">
    默认中国股票历史费率，佣金为示例值；收盘信号、次日开盘成交，费用、T+1
    与公司行动共用模拟盘引擎。更改参数需显式重新提交。
  </p>
  {#if failure}<StatusPanel title="回测提交失败" error={failure} />{/if}
</section>
