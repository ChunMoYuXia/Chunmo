import { expect, it } from 'vitest';
import { number, percent, percentInputToRatio, ratioInput } from './format';

it('does not render unknown values as zero', () => {
  expect(number(null)).toBe('—');
  expect(percent(null)).toBe('—');
  expect(number('0')).toBe('0.00');
});

it('turns human percentages into exact decimal DSL ratios', () => {
  expect(percentInputToRatio('0.7')).toBe('0.007');
  expect(percentInputToRatio('20')).toBe('0.2');
  expect(percentInputToRatio('-0.01')).toBe('-0.0001');
  expect(ratioInput('0.007')).toBe('0.7');
});
