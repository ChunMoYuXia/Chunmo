// @vitest-environment jsdom
import { expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import Funnel from './Funnel.svelte';

it('shows missing-value exclusion evidence and rolls back the condition prefix', async () => {
  const rollback = vi.fn();
  render(Funnel, {
    universeCount: 3,
    matchingCount: 1,
    onRollback: rollback,
    steps: [
      {
        index: 0,
        input_count: 3,
        remaining_count: 2,
        condition: { field: 'close', operator: 'gte', value: '10' },
        exclusions: { below_threshold: 1 },
      },
      {
        index: 1,
        input_count: 2,
        remaining_count: 1,
        condition: { field: 'momentum_20d', operator: 'gte', value: '.1' },
        exclusions: { missing_value: 1 },
      },
    ],
  });
  expect(screen.getByText('数据缺失：1')).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: '撤回条件 2' }));
  expect(rollback).toHaveBeenCalledWith(1);
});
