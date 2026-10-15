const decimalFormat = new Intl.NumberFormat('zh-CN', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function number(value: string | number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return digits === 2
    ? decimalFormat.format(Number(value))
    : new Intl.NumberFormat('zh-CN', { maximumFractionDigits: digits }).format(Number(value));
}

export function percent(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : `${number(value * 100)}%`;
}

function shiftDecimal(input: string, places: number): string {
  const match = /^(-?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(input.trim());
  if (!match || Math.abs(Number(match[4] ?? 0)) > 100) throw new Error('请输入有效小数');
  const digits = match[2] + (match[3] ?? '');
  const point = match[2].length + Number(match[4] ?? 0) + places;
  const shifted =
    point <= 0
      ? `0.${'0'.repeat(-point)}${digits}`
      : point >= digits.length
        ? digits + '0'.repeat(point - digits.length)
        : `${digits.slice(0, point)}.${digits.slice(point)}`;
  const [whole, fraction = ''] = shifted.split('.');
  const integer = whole.replace(/^0+(?=\d)/, '');
  const decimals = fraction.replace(/0+$/, '');
  const value = `${integer}${decimals ? `.${decimals}` : ''}`;
  return `${value === '0' ? '' : match[1]}${value}`;
}

export function percentInputToRatio(input: string): string {
  return shiftDecimal(input, -2);
}

export function ratioInput(value: ScreeningValue): string {
  if (typeof value !== 'string' && typeof value !== 'number') return '';
  return shiftDecimal(String(value), 2);
}
type ScreeningValue = string | number | boolean | string[];
