// Derived, pre-computed data for one trace. Computed once per loaded trace so
// the canvas painters only read typed arrays during a redraw.
import type { ExpertLabel, TokenRecord, Viz } from './types';

export interface Analysis {
  viz: Viz;
  /** tokens, layers, experts */
  T: number;
  NL: number;
  NE: number;
  /** true when the model mixes layer kinds (e.g. linear + full attention) */
  hybrid: boolean;
  /** index of the first decode token, -1 if none */
  firstDecode: number;
  /** token index -> index into steps/next_candidates when the token ends a step, else -1 */
  stepEnd: Int32Array;
  /** token index -> index of the step that contains it */
  stepOf: Int32Array;
  /** layers x tokens: 1 - log10(rank+1)/4 of the emitted next token, NaN when missing */
  lens: Float32Array;
  /** layers x tokens: (attn + ffn write RMS) / residual RMS, NaN when missing */
  write: Float32Array;
  /** 99th percentile of write, the top of the write color scale */
  writeScale: number;
  usage: Usage | null;
  isFullAttention(layer: number): boolean;
  rec(layer: number, token: number): TokenRecord | undefined;
  label(layer: number, expert: number): ExpertLabel | undefined;
}

export interface Usage {
  /** layers x experts token counts */
  pre: Uint16Array;
  dec: Uint16Array;
  /** per layer: max(pre + dec) */
  rowMax: Uint16Array;
  /** per layer: number of tokens with routing data */
  routed: Uint16Array;
}

export function lensValue(rank: number | null | undefined): number {
  if (rank == null || rank < 0) return NaN;
  return Math.max(0, 1 - Math.log10(rank + 1) / 4);
}

export function analyse(viz: Viz): Analysis {
  const T = viz.tokens.length;
  const NL = viz.n_layers || viz.layers.length;
  const NE = viz.n_experts || 0;
  const layers = viz.layers;
  const kinds = new Set(layers.map((l) => l.kind));
  const hybrid = kinds.size > 1;

  const firstDecode = viz.tokens.findIndex((t) => t.phase === 'decode');
  const stepEnd = new Int32Array(T).fill(-1);
  const stepOf = new Int32Array(T).fill(-1);
  viz.steps.forEach((s, j) => {
    for (let i = s.first; i < s.first + s.n && i < T; i++) stepOf[i] = j;
    const last = s.first + s.n - 1;
    if (last >= 0 && last < T) stepEnd[last] = j;
  });

  const lens = new Float32Array(NL * T).fill(NaN);
  const write = new Float32Array(NL * T).fill(NaN);
  const writes: number[] = [];
  for (let L = 0; L < NL; L++) {
    const toks = layers[L]?.tokens;
    if (!toks) continue;
    for (const key in toks) {
      const i = +key;
      if (!(i >= 0 && i < T)) continue;
      const r = toks[key];
      lens[L * T + i] = lensValue(r.next_rank);
      const w = ((r.write?.attn ?? 0) + (r.write?.ffn ?? 0)) / (r.resid_rms || 1);
      if (Number.isFinite(w)) {
        write[L * T + i] = w;
        writes.push(w);
      }
    }
  }
  writes.sort((a, b) => a - b);
  const writeScale = writes.length ? writes[Math.min(writes.length - 1, Math.floor(writes.length * 0.99))] || 1 : 1;

  let usage: Usage | null = null;
  if (NE > 0) {
    const pre = new Uint16Array(NL * NE);
    const dec = new Uint16Array(NL * NE);
    const rowMax = new Uint16Array(NL);
    const routed = new Uint16Array(NL);
    for (let L = 0; L < NL; L++) {
      const toks = layers[L]?.tokens;
      if (!toks) continue;
      for (const key in toks) {
        const r = toks[key];
        if (!r.experts?.length) continue;
        const i = +key;
        const target = viz.tokens[i]?.phase === 'decode' ? dec : pre;
        routed[L]++;
        for (const e of r.experts) if (e >= 0 && e < NE) target[L * NE + e]++;
      }
      let m = 0;
      for (let e = 0; e < NE; e++) m = Math.max(m, pre[L * NE + e] + dec[L * NE + e]);
      rowMax[L] = m;
    }
    usage = { pre, dec, rowMax, routed };
  }

  const experts = viz.experts ?? {};
  return {
    viz,
    T,
    NL,
    NE,
    hybrid,
    firstDecode,
    stepEnd,
    stepOf,
    lens,
    write,
    writeScale,
    usage,
    isFullAttention: (L) => hybrid && layers[L]?.kind === 'attention',
    rec: (L, i) => layers[L]?.tokens[String(i)],
    label: (L, e) => experts[`${L}.${e}`],
  };
}

/** Two traces can be compared when their layer and expert grids match. */
export function comparable(a: { arch: string; n_layers: number; n_experts: number }, b: typeof a): boolean {
  return a.arch === b.arch && a.n_layers === b.n_layers && a.n_experts === b.n_experts;
}
