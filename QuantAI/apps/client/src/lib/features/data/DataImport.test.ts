// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { configureConnection } from '$lib/api/client';
import ImportPreview from './ImportPreview.svelte';
import DataImport from './DataImport.svelte';
import { invalidPreview } from './test-fixtures';

afterEach(() => vi.unstubAllGlobals());

it('shows the offending row and does not commit an invalid preview', async () => {
  const commit = vi.fn();
  render(ImportPreview, { preview: invalidPreview, onCommit: commit });
  expect(screen.getByText(/第 3 行/)).toBeVisible();
  const button = screen.getByRole('button', { name: '提交导入' });
  expect(button).toBeDisabled();
  await userEvent.click(button);
  expect(commit).not.toHaveBeenCalled();
});

it('uploads real form data and keeps import failure visible without claiming success', async () => {
  configureConnection({ url: '', token: '' });
  const imported = vi.fn();
  const requests: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    requests.push(url);
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBeInstanceOf(File);
    return Response.json({ data: invalidPreview });
  });
  render(DataImport, { onImported: imported });
  const user = userEvent.setup();
  await user.upload(
    screen.getByLabelText('行情 CSV'),
    new File(['代码,日期,开盘\n000001,2024-01-02,10'], 'quotes.csv', { type: 'text/csv' }),
  );
  await user.click(screen.getByRole('button', { name: '预检数据' }));
  expect(await screen.findByText(/第 3 行/)).toBeVisible();
  expect(screen.getByRole('button', { name: '提交导入' })).toBeDisabled();
  expect(imported).not.toHaveBeenCalled();
  expect(requests).toEqual(['/v1/datasets/import/preview']);
});
