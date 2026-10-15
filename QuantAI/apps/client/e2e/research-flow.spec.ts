import { test, expect } from './fixtures';

test('synthetic data, portfolio and paper ledger survive reload', async ({ page }) => {
  await page.goto('/data');
  await page.getByLabel('证券数量').fill('3');
  await page.getByLabel('交易日数量').fill('60');
  await page.getByRole('button', { name: '创建演示数据' }).click();
  await expect(page.getByText('数据已就绪', { exact: true })).toBeVisible();
  await page.reload();
  await expect(
    page.getByText('用于验证研究流程，不代表真实市场行情。', { exact: true }),
  ).toBeVisible();
  await page.getByRole('link', { name: '选股', exact: true }).click();
  await page.getByRole('button', { name: '运行筛选', exact: true }).click();
  await expect(page.getByRole('heading', { name: '筛选结果', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '保存并带入组合' }).click();
  await expect(page.getByRole('heading', { name: '组合配置', exact: true })).toBeVisible();
  await page.getByLabel('历史窗口 / 交易日').fill('20');
  await page.getByLabel('配置方法').selectOption('minimum_variance');
  await page.getByRole('button', { name: '计算目标权重' }).click();
  await expect(page.getByText(/20 个历史观测/)).toBeVisible();
  await page.getByRole('button', { name: '保存组合', exact: true }).click();
  await page.getByRole('link', { name: '回测此组合 →' }).click();
  await page.getByRole('button', { name: '提交回测', exact: true }).click();
  await expect(page.getByRole('heading', { name: '回测结果', exact: true })).toBeVisible();
  await page.getByRole('link', { name: '查看证据报告 →' }).click();
  await expect(page.getByRole('heading', { name: '规则分析报告', exact: true })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出 Markdown' }).click();
  expect((await download).suggestedFilename()).toMatch(/\.md$/);
  await page.getByRole('link', { name: '模拟盘', exact: true }).click();
  await page.getByRole('button', { name: '创建模拟账户' }).click();
  await page.getByRole('button', { name: '提交模拟订单' }).click();
  await expect(page.getByText('已提交', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '推进下一交易日' }).click();
  await expect(page.getByText('已成交', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText('已成交', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '核验审计链' }).click();
  await expect(page.getByText(/核验通过/)).toBeVisible();
});
