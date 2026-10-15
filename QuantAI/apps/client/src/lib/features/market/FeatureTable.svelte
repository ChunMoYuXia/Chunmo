<script lang="ts">
  import { resolve } from '$app/paths';
  import type { Feature } from '$lib/api/types';
  import { number, percent } from '$lib/utils/format';
  let {
    features,
    selectedIds = [],
    onToggle,
  }: {
    features: Feature[];
    selectedIds?: string[];
    onToggle?: (securityId: string) => void;
  } = $props();
  let page = $state(0);
  const pageSize = 50;
  let visible = $derived(features.slice(page * pageSize, (page + 1) * pageSize));
  $effect(() => {
    if (page * pageSize >= features.length) page = 0;
  });
</script>

<div class="table-scroll">
  <table>
    <caption class="sr-only">证券与历史特征</caption><thead
      ><tr
        >{#if onToggle}<th>纳入</th>{/if}<th>证券</th><th>行业</th><th class="numeric">收盘 / 元</th
        ><th class="numeric">20日动量</th><th class="numeric">年化波动</th><th>数据状态</th></tr
      ></thead
    ><tbody>
      {#each visible as feature (feature.security_id)}<tr>
          {#if onToggle}<td
              ><input
                type="checkbox"
                aria-label={`纳入 ${feature.name}`}
                checked={selectedIds.includes(feature.security_id)}
                onchange={() => onToggle?.(feature.security_id)}
              /></td
            >{/if}
          <td
            ><a
              href={resolve(`/market?security=${encodeURIComponent(feature.security_id)}`)}
              class="cell-main">{feature.name}</a
            ><small
              >{feature.code}
              {feature.exchange === 'XSHG'
                ? '沪'
                : feature.exchange === 'XSHE'
                  ? '深'
                  : '京'}</small
            ></td
          >
          <td>{feature.industry ?? '未分类'}</td><td class="numeric">{number(feature.close)}</td>
          <td
            class="numeric"
            class:gain={(feature.momentum_20d ?? 0) > 0}
            class:loss={(feature.momentum_20d ?? 0) < 0}
            >{(feature.momentum_20d ?? 0) > 0 ? '+' : ''}{percent(feature.momentum_20d)}</td
          >
          <td class="numeric">{percent(feature.volatility_20d)}</td><td
            ><span title={feature.warnings.join('\n')}
              >{feature.stale
                ? '行情缺口'
                : feature.missing_sessions.length
                  ? '窗口缺失'
                  : feature.sample_count < 21
                    ? '历史不足'
                    : feature.price_basis === 'raw'
                      ? '原始价格'
                      : feature.price_basis === 'adjusted'
                        ? '已复权'
                        : '复权缺失'}</span
            ></td
          >
        </tr>{/each}
    </tbody>
  </table>
</div>
{#if features.length > pageSize}<div class="result-footer">
    <span>第 {page + 1} / {Math.ceil(features.length / pageSize)} 页，共 {features.length} 只</span>
    <div class="actions">
      <button disabled={page === 0} onclick={() => (page -= 1)}>上一页</button><button
        disabled={(page + 1) * pageSize >= features.length}
        onclick={() => (page += 1)}>下一页</button
      >
    </div>
  </div>{/if}
