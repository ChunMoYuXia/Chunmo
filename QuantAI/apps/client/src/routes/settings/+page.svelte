<script lang="ts">
  import { onMount } from 'svelte';
  import { request } from '$lib/api/client';
  import type { components } from '$lib/api/schema';
  import { applyPreferences, preferences } from '$lib/state/preferences';
  import StatusPanel from '$lib/components/StatusPanel.svelte';
  let config = $state<components['schemas']['LocalConfig'] | null>(null);
  let aiConfig = $state<components['schemas']['AIConfig'] | null>(null);
  let failure = $state<unknown>(null);
  onMount(() => {
    request<components['schemas']['AIConfig']>('/v1/ai/config')
      .then((value) => (aiConfig = value))
      .catch((error) => (failure = error));
    request<components['schemas']['LocalConfig']>('/v1/config')
      .then((value) => (config = value))
      .catch((error) => (failure = error));
  });
</script>

<svelte:head><title>设置 · 组合研究台</title></svelte:head>
<div class="page-heading">
  <div>
    <h1>设置</h1>
    <p>设置研究工作台的显示方式与本地环境。</p>
  </div>
</div>
<section class="panel">
  <h2>涨跌颜色</h2>
  <p class="muted">图表与表格同步生效；数字符号始终保留。</p>
  <div class="choice-row">
    <label
      ><input
        type="radio"
        name="up-color"
        checked={$preferences.upColor === 'red'}
        onchange={() => applyPreferences({ upColor: 'red' })}
      />红涨绿跌（A股惯例）</label
    ><label
      ><input
        type="radio"
        name="up-color"
        checked={$preferences.upColor === 'green'}
        onchange={() => applyPreferences({ upColor: 'green' })}
      />绿涨红跌</label
    >
  </div>
</section>
<section class="panel">
  <h2>运行环境</h2>
  {#if config}<dl class="provenance">
      <dt>运行模式</dt>
      <dd>{config.mode === 'local' ? '本地单用户' : config.mode}</dd>
      <dt>实盘交易</dt>
      <dd>{config.real_trading_enabled ? '已启用' : '未启用'}</dd>
      <dt>数据位置</dt>
      <dd>保存在本机；请按 README 中的说明备份数据库与来源文件。</dd>
    </dl>{:else if !failure}<StatusPanel
      title="正在检查本地服务"
      loading
    />{/if}{#if failure}<StatusPanel error={failure} />{/if}
</section>

<section class="panel">
  <h2>可选 AI 接入</h2>
  <p>
    {aiConfig?.configured
      ? '已配置模型：' + aiConfig.model
      : '尚未配置模型；规则报告与本地研究仍可使用。'}
  </p>
  <p class="muted">在启动后端前设置本机环境变量，随后重启本地服务。密钥只在后端进程读取。</p>
  <dl class="provenance">
    <dt>PORTFOLIO_AI_BASE_URL</dt>
    <dd>兼容 Chat Completions 的 API 基础地址，包含 /v1；使用 HTTPS 或本机 HTTP。</dd>
    <dt>PORTFOLIO_AI_MODEL</dt>
    <dd>该端点提供的模型名称。</dd>
    <dt>PORTFOLIO_AI_API_KEY</dt>
    <dd>服务商密钥；不要写入前端、Git、报告或截图。</dd>
  </dl>
  <p class="context-note">
    仅显式发起请求才调用模型；输出上限 2048 tokens，总时限 60 秒。供应商需支持 max_completion_tokens
    和
    stream_options.include_usage；不兼容时会返回错误。取消或中断可能无法收到最终用量，费用以服务商账单为准。
  </p>
</section>
