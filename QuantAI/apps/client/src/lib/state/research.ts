import { get, writable } from 'svelte/store';
import { request } from '$lib/api/client';
import type { Dataset } from '$lib/api/types';

export type ResearchContext = { datasetId: string; asOf: string };
export const researchContext = writable<ResearchContext>({ datasetId: '', asOf: '' });
export const datasetCatalog = writable<Dataset[]>([]);

export function marketQuery(context: ResearchContext, extra: Record<string, string> = {}): string {
  return new URLSearchParams({
    dataset_id: context.datasetId,
    as_of: context.asOf,
    ...extra,
  }).toString();
}

export function setResearchContext(context: ResearchContext): void {
  researchContext.set(context);
  try {
    localStorage.setItem('research-context-v1', JSON.stringify(context));
  } catch {
    /* Research still works without browser storage. */
  }
}

export function chooseDataset(dataset: Dataset, day = dataset.last_session): void {
  setResearchContext({
    datasetId: dataset.dataset_id,
    asOf: new Date(`${day}T15:00:00+08:00`).toISOString(),
  });
}

export async function refreshDatasets(signal?: AbortSignal): Promise<Dataset[]> {
  const datasets = await request<Dataset[]>('/v1/datasets', { signal });
  datasetCatalog.set(datasets);
  const current = get(researchContext);
  if (!datasets.some((dataset) => dataset.dataset_id === current.datasetId)) {
    let stored: Partial<ResearchContext> = {};
    try {
      stored = JSON.parse(localStorage.getItem('research-context-v1') ?? '{}');
    } catch {
      /* Ignore invalid preferences. */
    }
    const previous = datasets.find((dataset) => dataset.dataset_id === stored.datasetId);
    if (previous && stored.asOf && Number.isFinite(Date.parse(stored.asOf)))
      setResearchContext({ datasetId: previous.dataset_id, asOf: stored.asOf });
    else if (datasets[0]) chooseDataset(datasets[0]);
    else researchContext.set({ datasetId: '', asOf: '' });
  }
  return datasets;
}
