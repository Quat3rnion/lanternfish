// Types for the lanternfish HTTP API. Keep in sync with the Rust server.

export type Phase = 'prefill' | 'decode';
export type LayerKind = 'attention' | 'linear' | 'unknown';

export interface TraceSummary {
  name: string;
  model: { name: string; arch: string; n_layers: number; n_experts: number };
  tokens: number;
  created: string;
}

export interface Token {
  i: number;
  piece: string;
  phase: Phase;
  id: number;
}

export interface Step {
  first: number;
  n: number;
  phase: Phase;
}

export interface Candidate {
  id: number;
  piece: string;
  p: number;
}

export interface TokenRecord {
  lens: [string, number][];
  resid_rms: number;
  write: { attn?: number | null; ffn?: number | null };
  next_rank?: number | null;
  experts?: number[];
  weights?: number[];
  shared_gate?: number | null;
  streams_rms?: number[];
}

export interface Layer {
  layer: number;
  kind: LayerKind;
  tokens: Record<string, TokenRecord>;
}

export interface ExpertLabel {
  pos: string[];
  neg: string[];
  rank1: number;
}

export interface Viz {
  name: string;
  model: {
    arch: string;
    name: string;
    n_layers: number;
    hidden?: number;
    n_experts: number;
    top_k: number;
    [k: string]: unknown;
  };
  n_layers: number;
  n_experts: number;
  top_k: number;
  tokens: Token[];
  steps: Step[];
  next_candidates: Candidate[][];
  layers: Layer[];
  experts?: Record<string, ExpertLabel>;
  lens_last_layer_agreement?: number | null;
  lens_note?: string;
  prompt?: string;
  created?: string;
}

export interface ModelFile {
  path: string;
  name: string;
  size_bytes: number;
}

export type RunState = 'queued' | 'running' | 'done' | 'failed';

export interface Run {
  id: string;
  name: string;
  model: string;
  state: RunState;
  phase: string;
  progress: number;
  log_tail: string;
  error?: string;
}

export interface RunRequest {
  model: string;
  name: string;
  prompt: string;
  n: number;
  raw: boolean;
}

export type ViewId = 'route' | 'lens' | 'write' | 'usage';
export type CompareMode = 'side' | 'diff';
