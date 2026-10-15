// @vitest-environment jsdom
import { expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import WeightEditor from './WeightEditor.svelte';

it('preserves a decimal point while the controlled parent echoes parsed weights', async () => {
  let changed: Record<string, string> = {};
  const view = render(WeightEditor, {
    weights: { 'XSHG:600001': '0.1' },
    onChange: (value) => (changed = value),
  });
  const input = screen.getByLabelText('XSHG:600001 目标权重 / %');
  const user = userEvent.setup();
  await user.clear(input);
  await user.type(input, '12.');
  await view.rerender({ weights: changed, onChange: (value) => (changed = value) });
  expect(input).toHaveValue('12.');
  await user.type(input, '345');
  expect(changed).toEqual({ 'XSHG:600001': '0.12345' });
});

it('emits decimal ratios without rounding drift or client-side optimization', async () => {
  let changed: Record<string, string> = {};
  render(WeightEditor, {
    weights: { '600001.SH': '0.007' },
    onChange: (value) => (changed = value),
  });
  const input = screen.getByLabelText('600001.SH 目标权重 / %');
  expect(input).toHaveValue('0.7');
  const user = userEvent.setup();
  await user.clear(input);
  await user.type(input, '12.345');
  expect(changed).toEqual({ '600001.SH': '0.12345' });
});
