import { init, use, type ECharts } from 'echarts/core';
import { CandlestickChart, BarChart, LineChart, PieChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  LegendComponent,
  AriaComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

use([
  CandlestickChart,
  BarChart,
  LineChart,
  PieChart,
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  LegendComponent,
  AriaComponent,
  CanvasRenderer,
]);
export function initChart(element: HTMLElement): ECharts {
  return init(element, undefined, {
    renderer: 'canvas',
    devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2),
  });
}
