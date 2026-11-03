// Token x layer heatmaps: logit lens, write energy, and the lens difference.
import type { Analysis } from '../analysis';
import { C, div, seq } from '../colors';
import { int, vis } from '../format';
import { font, type Hit, type Painter, type Tip } from './painter';
import { layerName, recordLines } from './tips';

const LEFT = 52;
const TOP = 10;

interface GridSpec {
  T: number;
  NL: number;
  sel: number;
  firstDecode: number;
  fullAttention: (L: number) => boolean;
  /** fill color for a cell or null when there is no data */
  color: (L: number, i: number) => string | null;
  /** columns to mark with a red tick above the grid */
  mismatch?: (i: number) => boolean;
  tip: (L: number, i: number) => Tip;
  alt: string;
}

function cellHeight(NL: number): number {
  return Math.max(8, Math.min(16, Math.floor(540 / NL)));
}

export function gridPainter(g: GridSpec): Painter {
  const ch = cellHeight(g.NL);
  const cwFor = (w: number) => Math.max(3, Math.min(18, (w - LEFT - 6) / Math.max(1, g.T)));
  return {
    alt: g.alt,
    size(w) {
      const cw = cwFor(w);
      return { width: Math.ceil(LEFT + cw * g.T + 6), height: TOP + g.NL * ch + 38 };
    },
    draw(ctx, W) {
      const cw = cwFor(W);
      const gap = cw > 5 ? 1 : 0;
      const labelEvery = g.NL <= 16 ? 1 : ch >= 14 ? 2 : 4;
      ctx.font = font(10);
      ctx.textBaseline = 'middle';
      for (let L = 0; L < g.NL; L++) {
        const y = TOP + L * ch;
        // diamonds sit in their own column so they never collide with the layer labels
        if (g.fullAttention(L)) {
          ctx.fillStyle = C.dec;
          ctx.fillText('\u25c6', 2, y + ch / 2);
        }
        if (L % labelEvery === 0 || L === g.NL - 1) {
          ctx.fillStyle = C.dim;
          ctx.fillText('L' + L, 14, y + ch / 2);
        }
        for (let i = 0; i < g.T; i++) {
          ctx.fillStyle = g.color(L, i) ?? C.empty;
          ctx.fillRect(LEFT + i * cw, y, cw - gap, ch - 1);
        }
      }
      ctx.textBaseline = 'alphabetic';
      const bottom = TOP + g.NL * ch;
      if (g.mismatch) {
        ctx.fillStyle = C.mismatch;
        for (let i = 0; i < g.T; i++) if (g.mismatch(i)) ctx.fillRect(LEFT + i * cw, 2, Math.max(1, cw - gap), 5);
      }
      // token index ticks
      ctx.fillStyle = C.dim;
      ctx.font = font(10);
      const every = cw >= 12 ? 5 : cw >= 6 ? 10 : 20;
      for (let i = 0; i < g.T; i += every) {
        ctx.fillRect(LEFT + i * cw, bottom + 1, 1, 3);
        ctx.fillText(String(i), LEFT + i * cw + 2, bottom + 13);
      }
      if (g.firstDecode > 0 && g.firstDecode < g.T) {
        const x = LEFT + g.firstDecode * cw - 1;
        ctx.strokeStyle = C.dec;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, TOP - 2);
        ctx.lineTo(x, bottom + 4);
        ctx.stroke();
        ctx.fillStyle = C.dec;
        ctx.fillText('decode \u2192', x + 3, bottom + 28);
        ctx.fillStyle = C.pre;
        ctx.textAlign = 'right';
        ctx.fillText('\u2190 prefill', x - 3, bottom + 28);
        ctx.textAlign = 'left';
      }
      if (g.sel >= 0 && g.sel < g.T) {
        ctx.strokeStyle = C.acc;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(LEFT + g.sel * cw - 0.75, TOP - 1, cw + 0.5, g.NL * ch + 1);
      }
      ctx.lineWidth = 1;
      return (x, y): Hit | null => {
        const i = Math.floor((x - LEFT) / cw);
        if (i < 0 || i >= g.T) return null;
        const L = Math.floor((y - TOP) / ch);
        if (L < 0 || L >= g.NL) {
          if (y > bottom) return { tip: { title: `token #${i}`, lines: [{ text: 'click to select', tone: 'dim' }] }, token: i };
          return null;
        }
        return { tip: g.tip(L, i), token: i };
      };
    },
  };
}

export function lensPainter(a: Analysis, sel: number): Painter {
  const { T, NL } = a;
  return gridPainter({
    T, NL, sel,
    firstDecode: a.firstDecode,
    fullAttention: a.isFullAttention,
    color: (L, i) => {
      const v = a.lens[L * T + i];
      return Number.isNaN(v) ? null : seq(v);
    },
    tip: (L, i) => ({
      title: `#${i} \u00b7 ${layerName(a, L)}`,
      titleMono: vis(a.viz.tokens[i].piece),
      lines: recordLines(a, L, i),
    }),
    alt: `Logit lens heatmap, ${T} tokens by ${NL} layers. Selected token ${sel}.`,
  });
}

export function writePainter(a: Analysis, sel: number): Painter {
  const { T, NL } = a;
  return gridPainter({
    T, NL, sel,
    firstDecode: a.firstDecode,
    fullAttention: a.isFullAttention,
    color: (L, i) => {
      const v = a.write[L * T + i];
      return Number.isNaN(v) ? null : seq(v / a.writeScale);
    },
    tip: (L, i) => {
      const v = a.write[L * T + i];
      return {
        title: `#${i} \u00b7 ${layerName(a, L)}`,
        titleMono: vis(a.viz.tokens[i].piece),
        lines: [
          { label: 'write / resid', text: Number.isNaN(v) ? 'n/a' : v.toFixed(3), tone: 'acc' },
          ...recordLines(a, L, i),
        ],
      };
    },
    alt: `Write energy heatmap, ${T} tokens by ${NL} layers. Selected token ${sel}.`,
  });
}

/** Difference of lens values, A minus B, aligned by token position. */
export function lensDiffPainter(a: Analysis, b: Analysis, sel: number): Painter & { scale: number } {
  const T = Math.min(a.T, b.T);
  const NL = a.NL;
  const vals = new Float32Array(NL * T).fill(NaN);
  const mags: number[] = [];
  for (let L = 0; L < NL; L++)
    for (let i = 0; i < T; i++) {
      const d = a.lens[L * a.T + i] - b.lens[L * b.T + i];
      vals[L * T + i] = d;
      if (!Number.isNaN(d)) mags.push(Math.abs(d));
    }
  mags.sort((x, y) => x - y);
  const scale = Math.max(0.05, mags.length ? mags[Math.floor((mags.length - 1) * 0.99)] : 1);
  const rank = (x: Analysis, L: number, i: number) => {
    const r = x.rec(L, i)?.next_rank;
    return r == null ? 'n/a' : int(r);
  };
  const painter = gridPainter({
    T, NL, sel,
    firstDecode: a.firstDecode === b.firstDecode ? a.firstDecode : -1,
    fullAttention: a.isFullAttention,
    color: (L, i) => {
      const d = vals[L * T + i];
      return Number.isNaN(d) ? null : div(d / scale);
    },
    mismatch: (i) => a.viz.tokens[i].id !== b.viz.tokens[i].id,
    tip: (L, i) => {
      const d = vals[L * T + i];
      const ta = a.viz.tokens[i], tb = b.viz.tokens[i];
      return {
        title: `#${i} \u00b7 ${layerName(a, L)}`,
        lines: [
          { label: 'A token', text: vis(ta.piece), mono: true, tone: 'a' },
          { label: 'B token', text: vis(tb.piece), mono: true, tone: ta.id === tb.id ? 'b' : 'warn' },
          { label: 'next-token rank', text: `A ${rank(a, L, i)}  \u00b7  B ${rank(b, L, i)}` },
          { label: 'lens value A \u2212 B', text: Number.isNaN(d) ? 'n/a' : (d >= 0 ? '+' : '') + d.toFixed(3) },
        ],
      };
    },
    alt: `Logit lens difference between two traces, ${T} aligned tokens by ${NL} layers.`,
  });
  return Object.assign(painter, { scale });
}
