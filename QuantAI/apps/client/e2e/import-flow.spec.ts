import { test, expect } from './fixtures';

function csvFixture(invalid = false) {
  const days: string[] = [];
  const date = new Date('2024-01-02T00:00:00Z');
  while (days.length < 65) {
    if (date.getUTCDay() !== 0 && date.getUTCDay() !== 6)
      days.push(date.toISOString().slice(0, 10));
    date.setUTCDate(date.getUTCDate() + 1);
  }
  const rows = ['代码,日期,开盘,最高,最低,收盘,成交量,涨停价,跌停价'];
  for (const code of ['600001', '000002', '600003'])
    days.forEach((day, index) => {
      const price = (10 + index * 0.01).toFixed(2);
      rows.push(
        `${code},${day},${price},${invalid && index === 1 && code === '600001' ? '1' : price},${price},${price},1000000,20,5`,
      );
    });
  return {
    quotes: Buffer.from('\ufeff' + rows.join('\n'), 'utf8'),
    calendar: Buffer.from('date\n' + days.join('\n'), 'utf8'),
  };
}

test('CSV preflight rejects the offending row, then imported research completes with source evidence', async ({
  page,
}) => {
  await page.goto('/data');
  const invalid = csvFixture(true);
  await page
    .getByLabel('行情 CSV', { exact: true })
    .setInputFiles({ name: 'quotes.csv', mimeType: 'text/csv', buffer: invalid.quotes });
  await page
    .getByLabel('交易日历 CSV', { exact: true })
    .setInputFiles({ name: 'calendar.csv', mimeType: 'text/csv', buffer: invalid.calendar });
  await page.getByRole('button', { name: '预检数据' }).click();
  await expect(page.getByText(/第 3 行/)).toBeVisible();
  await expect(page.getByRole('button', { name: '提交导入' })).toBeDisabled();
  const valid = csvFixture();
  await page
    .getByLabel('行情 CSV', { exact: true })
    .setInputFiles({ name: 'quotes.csv', mimeType: 'text/csv', buffer: valid.quotes });
  await page.getByRole('button', { name: '预检数据' }).click();
  await page.getByRole('button', { name: '提交导入' }).click();
  await expect(page.getByText(/已提交 195 条记录/)).toBeVisible();
  await page.getByRole('link', { name: '选股', exact: true }).click();
  await page.getByRole('button', { name: '运行筛选', exact: true }).click();
  await page.getByRole('button', { name: '保存并带入组合' }).click();
  await page.getByLabel('历史窗口 / 交易日').fill('20');
  await page.getByRole('button', { name: '计算目标权重' }).click();
  await expect(page.getByLabel('XSHG:600001 目标权重 / %')).not.toHaveValue('0');
  await page.getByRole('button', { name: '保存组合', exact: true }).click();
  await page.getByRole('link', { name: '回测此组合 →' }).click();
  await page.getByRole('button', { name: '提交回测', exact: true }).click();
  await expect(page.getByRole('heading', { name: '回测结果', exact: true })).toBeVisible();
  await page.getByRole('link', { name: '查看证据报告 →' }).click();
  await expect(
    page.getByText('本地导入数据；数据质量和授权由使用者核验。', { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: '规则分析报告', exact: true })).toBeVisible();
});
