import type { ModelFile, Run, RunRequest, TraceSummary, Viz } from './types';

// Relative URLs: the app is served at "/" by the Rust server and by Vite.
const BASE = 'api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/${path}`, init);
  } catch (e) {
    throw new Error(`network error: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const text = await res.text();
      try {
        const j = JSON.parse(text);
        if (j && typeof j.error === 'string') msg = `${res.status}: ${j.error}`;
        else if (text) msg = `${res.status}: ${text.slice(0, 200)}`;
      } catch {
        if (text) msg = `${res.status}: ${text.slice(0, 200)}`;
      }
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const api = {
  traces: () => request<TraceSummary[]>('traces'),
  trace: (name: string) => request<Viz>(`traces/${encodeURIComponent(name)}`),
  models: () => request<ModelFile[]>('models'),
  runs: () => request<Run[]>('runs'),
  startRun: (body: RunRequest) =>
    request<{ id: string }>('runs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),
};
