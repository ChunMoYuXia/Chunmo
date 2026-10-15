import type { Preview } from '$lib/api/types';

export const invalidPreview: Preview = {
  id: 'p1',
  name: '导入行情',
  row_count: 0,
  source_kind: 'imported',
  content_hash: 'abc',
  rows: [],
  errors: [{ row: 3, code: 'INVALID_OHLC', message: '最高价小于开盘价' }],
  warnings: [],
  calendar_sessions: 2,
  volume_unit: 'shares',
  can_commit: false,
};
