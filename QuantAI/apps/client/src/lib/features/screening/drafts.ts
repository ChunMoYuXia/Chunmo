import type { FieldDefinition, ScreeningFilter } from '$lib/api/types';
import { percentInputToRatio, ratioInput } from '$lib/utils/format';

export type ConditionDraft = Pick<ScreeningFilter, 'field' | 'operator'> & {
  key: string;
  displayValue: string;
};
let sequence = 0;

export function newCondition(field: FieldDefinition): ConditionDraft {
  return {
    key: String(++sequence),
    field: field.key,
    operator: field.data_type === 'number' ? 'gte' : 'eq',
    displayValue: field.data_type === 'boolean' ? 'false' : field.data_type === 'number' ? '0' : '',
  };
}

export function filtersFromDrafts(
  drafts: ConditionDraft[],
  fields: FieldDefinition[],
): ScreeningFilter[] {
  return drafts.map((draft) => {
    const definition = fields.find((field) => field.key === draft.field);
    if (!definition) throw new Error('指标定义尚未加载');
    let value: ScreeningFilter['value'] = draft.displayValue;
    if (definition.data_type === 'number') {
      if (!draft.displayValue.trim() || !Number.isFinite(Number(draft.displayValue)))
        throw new Error(`${definition.label}需要有效数字`);
      value =
        definition.display_scale === 100
          ? percentInputToRatio(draft.displayValue)
          : draft.displayValue;
    } else if (definition.data_type === 'boolean') value = draft.displayValue === 'true';
    else if (draft.operator === 'in') {
      value = draft.displayValue
        .split(/[,，]/)
        .map((item) => item.trim())
        .filter(Boolean);
      if (!value.length) throw new Error(`${definition.label}需要至少一个文本值`);
    }
    return { field: draft.field, operator: draft.operator, value };
  });
}

export function draftsFromFilters(
  filters: ScreeningFilter[],
  fields: FieldDefinition[],
): ConditionDraft[] {
  return filters.map((filter) => {
    const field = fields.find((item) => item.key === filter.field);
    const displayValue =
      field?.display_scale === 100
        ? ratioInput(filter.value)
        : Array.isArray(filter.value)
          ? filter.value.join('，')
          : String(filter.value);
    return {
      key: String(++sequence),
      field: filter.field,
      operator: filter.operator,
      displayValue,
    };
  });
}
