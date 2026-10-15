import { writable } from 'svelte/store';

export type Preferences = { upColor: 'red' | 'green' };
export const preferences = writable<Preferences>({ upColor: 'red' });

export function applyPreferences(value: Preferences): void {
  preferences.set(value);
  if (typeof document !== 'undefined') document.documentElement.dataset.upColor = value.upColor;
  try {
    localStorage.setItem('display-preferences-v1', JSON.stringify(value));
  } catch {
    /* Optional local preference. */
  }
}

export function initializePreferences(): void {
  try {
    const stored = JSON.parse(localStorage.getItem('display-preferences-v1') ?? '{}');
    if (stored.upColor === 'red' || stored.upColor === 'green')
      applyPreferences({ upColor: stored.upColor });
  } catch {
    /* Use the default for invalid preferences. */
  }
}
