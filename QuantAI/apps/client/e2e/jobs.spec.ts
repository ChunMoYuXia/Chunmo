import { test, expect } from './fixtures';

test('long research can be cancelled and invalid ranges expose actionable failures', async ({
  page,
}) => {
  await page.goto('/data');
  await page.getByLabel('证券数量').fill('200');
  await page.getByLabel('交易日数量').fill('756');
  await page.getByRole('button', { name: '创建演示数据' }).click();
  await expect(page.getByText('数据已就绪', { exact: true })).toBeVisible({ timeout: 60000 });
  await page.getByRole('link', { name: '组合', exact: true }).click();
  await page.getByRole('button', { name: '全选', exact: true }).click();
  await page.getByRole('button', { name: '计算目标权重' }).click();
  await expect(page.getByLabel('XSHG:600001 目标权重 / %')).not.toHaveValue('0');
  await page.getByRole('button', { name: '保存组合', exact: true }).click();
  await page.getByRole('link', { name: '回测此组合 →' }).click();
  await page.getByRole('button', { name: '提交回测', exact: true }).click();
  await page.getByRole('button', { name: '取消后台任务' }).click();
  await expect(page.getByText('任务已取消', { exact: true })).toBeVisible();
  const firstDay = await page.getByLabel('开始日期', { exact: true }).inputValue();
  await page.getByLabel('结束日期', { exact: true }).fill(firstDay);
  await page.getByRole('button', { name: '提交回测', exact: true }).click();
  await expect(page.getByText(/INSUFFICIENT_HISTORY/)).toBeVisible();
  await expect(page.getByText(/诊断编号/)).toBeVisible();
});
