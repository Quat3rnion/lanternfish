// Palette and color scales. CSS custom properties in app.css use the same values.

export const C = {
  bg: '#0f1115',
  panel: '#171a21',
  empty: '#1b1e25',
  line: '#2a2f3a',
  fg: '#d8dde6',
  dim: '#8b93a3',
  pre: '#5b8def',
  dec: '#f0a04b',
  acc: '#56d39a',
  prev: 'rgba(150,156,170,0.55)',
  mismatch: '#e5534b',
  a: '#f28fdc',
  b: '#62d8f5',
} as const;

const VIRIDIS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'];

function hexRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lut(stops: string[], size = 256): string[] {
  const rgb = stops.map(hexRgb);
  const out: string[] = [];
  for (let i = 0; i < size; i++) {
    const t = (i / (size - 1)) * (rgb.length - 1);
    const k = Math.min(rgb.length - 2, Math.floor(t));
    const f = t - k;
    const c = rgb[k].map((v, j) => Math.round(v + (rgb[k + 1][j] - v) * f));
    out.push(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  return out;
}

const SEQ = lut(VIRIDIS);
// Diverging: cyan (B larger) <- dark neutral -> pink (A larger). Dark center fits the dark
// theme; the hues avoid the prefill blue, decode orange and selection green.
const DIV_STOPS = ['#62d8f5', '#2b7d96', '#232838', '#9c4689', '#f28fdc'];
const DIV = lut(DIV_STOPS, 257);

/** Sequential scale, t in 0..1. */
export function seq(t: number): string {
  const k = Math.round(Math.max(0, Math.min(1, t)) * 255);
  return SEQ[k];
}

/** Diverging scale, t in -1..1. */
export function div(t: number): string {
  const k = Math.round((Math.max(-1, Math.min(1, t)) + 1) * 128);
  return DIV[k];
}

export const SEQ_CSS = `linear-gradient(to right, ${VIRIDIS.join(', ')})`;
export const DIV_CSS = `linear-gradient(to right, ${DIV_STOPS.join(', ')})`;
