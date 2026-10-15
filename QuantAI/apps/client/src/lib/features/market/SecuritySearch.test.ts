// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import SecuritySearch from './SecuritySearch.svelte';

afterEach(() => vi.unstubAllGlobals());
const context = { datasetId: 'd1', asOf: '2024-01-02T07:00:00Z' };
function security(name: string, code: string) {
  return {
    security_id: `XSHE:${code}`,
    code,
    exchange: 'XSHE',
    name,
    industry: '科技',
    initials: 'ys',
    board: 'main',
    currency: 'CNY',
  };
}

it('supports keyboard selection in actual search results', async () => {
  vi.stubGlobal('fetch', async () =>
    Response.json({ data: [security('演示金融', '000001'), security('演示消费', '000002')] }),
  );
  const select = vi.fn();
  render(SecuritySearch, { props: { context, onSelect: select } });
  const user = userEvent.setup();
  await user.type(screen.getByRole('combobox', { name: '搜索股票' }), '演示');
  await screen.findByRole('option', { name: /演示消费/ });
  await user.keyboard('{ArrowDown}{Enter}');
  expect(select.mock.calls[0][0].security_id).toBe('XSHE:000002');
});

it('discards stale results after a dataset switch even if the old response ignores abort', async () => {
  let finishOld: (value: Response) => void = () => {};
  vi.stubGlobal('fetch', (url: string) =>
    url.includes('dataset_id=d1')
      ? new Promise<Response>((resolve) => {
          finishOld = resolve;
        })
      : Promise.resolve(Response.json({ data: [security('新数据', '000002')] })),
  );
  const view = render(SecuritySearch, { props: { context, onSelect: vi.fn() } });
  await userEvent.type(screen.getByRole('combobox'), '数据');
  await screen.findByText('搜索中…');
  await view.rerender({ context: { ...context, datasetId: 'd2' }, onSelect: vi.fn() });
  expect(await screen.findByRole('option', { name: /新数据/ })).toBeVisible();
  finishOld(Response.json({ data: [security('旧数据', '000001')] }));
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(screen.queryByRole('option', { name: /旧数据/ })).not.toBeInTheDocument();
});
