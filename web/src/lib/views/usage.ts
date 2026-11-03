// Expert usage: per layer, how many tokens each expert received.
import type { Analysis } from '../analysis';
import { C, div } from '../colors';
import { pct } from '../format';
import { font, type Hit, type Painter } from './painter';
import { expertLines } from './tips';

const ROW = 15;
const LEFT = 48;
const RIGHT = 10;
const AXIS = 22;

function labels(ctx: CanvasRenderingContext2D, a: Analysis, rowBottom: (L: number) => number) {
  ctx.font = font(10);
  for (let L = 0; L < a.NL; L++) {
    const fa = a.isFullAttention(L);
    ctx.fillStyle = fa ? C.dec : C.dim;
    ctx.fillText((fa ? '\u25c6' : ' ') + 'L' + L, 2, rowBottom(L) - 2);
  }
}

function axis(ctx: CanvasRenderingContext2D, a: Analysis, pw: number) {
  ctx.fillStyle = C.dim;
  ctx.font = font(10);
  ctx.textAlign = 'center';
  for (const e of [0, a.NE >> 2, a.NE >> 1, (3 * a.NE) >> 2, a.NE - 1])
    ctx.fillText(String(e), LEFT + ((e + 0.5) / a.NE) * pw, a.NL * ROW + 14);
  ctx.textAlign = 'left';
}

/** Find the expert under x, snapping to the nearest non-empty expert within 3 px. */
function pick(x: number, pw: number, NE: number, has: (e: number) => boolean): number {
  const e0 = Math.floor(((x - LEFT) / pw) * NE);
  const span = Math.max(1, Math.ceil((3 / pw) * NE));
  for (let d = 0; d <= span; d++) {
    for (const e of d ? [e0 - d, e0 + d] : [e0]) if (e >= 0 && e < NE && has(e)) return e;
  }
  return -1;
}

export function usagePainter(a: Analysis, sel: number): Painter {
  const u = a.usage!;
  const { NL, NE } = a;
  return {
    alt: `Expert usage over ${a.T} tokens: ${NL} layers by ${NE} experts.`,
    size: (w) => ({ width: Math.max(w, 400), height: NL * ROW + AXIS }),
    draw(ctx, W) {
      const pw = W - LEFT - RIGHT;
      const bw = Math.max(1, pw / NE);
      const yb = (L: number) => L * ROW + ROW - 3;
      labels(ctx, a, (L) => L * ROW + ROW - 1);
      for (let L = 0; L < NL; L++) {
        const y0 = yb(L);
        const m = Math.max(1, u.rowMax[L]);
        ctx.fillStyle = 'rgba(255,255,255,0.035)';
        ctx.fillRect(LEFT, y0, pw, 1);
        for (let e = 0; e < NE; e++) {
          const cp = u.pre[L * NE + e], cd = u.dec[L * NE + e];
          if (!cp && !cd) continue;
          const x = LEFT + (e / NE) * pw;
          const hp = ((ROW - 4) * cp) / m, hd = ((ROW - 4) * cd) / m;
          if (cp) {
            ctx.fillStyle = C.pre;
            ctx.fillRect(x, y0 - hp, bw, hp);
          }
          if (cd) {
            ctx.fillStyle = C.dec;
            ctx.fillRect(x, y0 - hp - hd, bw, hd);
          }
        }
        // the selected token's experts: accent ticks under the baseline
        const r = a.rec(L, sel);
        if (r?.experts) {
          ctx.fillStyle = C.acc;
          for (const e of r.experts) ctx.fillRect(LEFT + (e / NE) * pw - 0.5, y0 + 1, Math.max(2, bw), 2);
        }
      }
      axis(ctx, a, pw);
      return (x, y): Hit | null => {
        const L = Math.floor(y / ROW);
        if (L < 0 || L >= NL || x < LEFT || x > LEFT + pw) return null;
        const selExperts = new Set(a.rec(L, sel)?.experts ?? []);
        const e = pick(x, pw, NE, (e) => u.pre[L * NE + e] + u.dec[L * NE + e] > 0 || selExperts.has(e));
        if (e < 0) return null;
        const cp = u.pre[L * NE + e], cd = u.dec[L * NE + e];
        return {
          tip: {
            title: `L${L} expert ${e}`,
            lines: [
              { text: `${cp} prefill + ${cd} decode tokens of ${u.routed[L]}`, tone: 'dim' },
              ...(selExperts.has(e) ? [{ text: `used by the selected token #${sel}`, tone: 'acc' as const }] : []),
              ...expertLines(a, L, e),
            ],
          },
        };
      };
    },
  };
}

/** Difference of routing share (tokens routed to expert / routed tokens), A minus B. */
export function usageDiffPainter(a: Analysis, b: Analysis): Painter & { scale: number } {
  const ua = a.usage!, ub = b.usage!;
  const { NL, NE } = a;
  const d = new Float32Array(NL * NE);
  let scale = 0;
  for (let L = 0; L < NL; L++) {
    const na = Math.max(1, ua.routed[L]), nb = Math.max(1, ub.routed[L]);
    for (let e = 0; e < NE; e++) {
      const k = L * NE + e;
      const v = (ua.pre[k] + ua.dec[k]) / na - (ub.pre[k] + ub.dec[k]) / nb;
      d[k] = v;
      scale = Math.max(scale, Math.abs(v));
    }
  }
  scale = scale || 1;
  return {
    scale,
    alt: `Expert usage difference between two traces: ${NL} layers by ${NE} experts.`,
    size: (w) => ({ width: Math.max(w, 400), height: NL * ROW + AXIS }),
    draw(ctx, W) {
      const pw = W - LEFT - RIGHT;
      const bw = Math.max(1, pw / NE);
      labels(ctx, a, (L) => L * ROW + ROW - 1);
      ctx.fillStyle = C.panel;
      ctx.fillRect(LEFT, 0, pw, NL * ROW);
      for (let L = 0; L < NL; L++) {
        const y = L * ROW + 1;
        for (let e = 0; e < NE; e++) {
          const v = d[L * NE + e];
          if (!v) continue;
          ctx.fillStyle = div(v / scale);
          ctx.fillRect(LEFT + (e / NE) * pw, y, bw, ROW - 2);
        }
      }
      axis(ctx, a, pw);
      return (x, y): Hit | null => {
        const L = Math.floor(y / ROW);
        if (L < 0 || L >= NL || x < LEFT || x > LEFT + pw) return null;
        const e = pick(x, pw, NE, (e) => d[L * NE + e] !== 0);
        if (e < 0) return null;
        const k = L * NE + e;
        const ca = ua.pre[k] + ua.dec[k], cb = ub.pre[k] + ub.dec[k];
        const sa = ca / Math.max(1, ua.routed[L]), sb = cb / Math.max(1, ub.routed[L]);
        return {
          tip: {
            title: `L${L} expert ${e}`,
            lines: [
              { label: 'A', text: `${ca} of ${ua.routed[L]} tokens (${pct(sa, 1)})`, tone: 'a' },
              { label: 'B', text: `${cb} of ${ub.routed[L]} tokens (${pct(sb, 1)})`, tone: 'b' },
              { label: 'A \u2212 B', text: `${d[k] >= 0 ? '+' : ''}${(d[k] * 100).toFixed(1)} points` },
              ...expertLines(a, L, e),
            ],
          },
        };
      };
    },
  };
}
