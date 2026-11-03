// Expert route: one row per layer, x = routed expert id. Dots for the selected
// token (accent) and the previous token (grey), dot area ~ router weight.
import type { Analysis } from '../analysis';
import { C } from '../colors';
import { clip, pct, vis } from '../format';
import { font, type Hit, type Painter } from './painter';
import { expertLines, layerName, recordLines } from './tips';

const ROW = 15;
const LEFT = 72;
const RIGHT = 150;
const AXIS = 26;

function radius(w: number | undefined): number {
  return 1.5 + 6 * Math.sqrt(w ?? 0.1);
}

export function routePainter(a: Analysis, sel: number): Painter {
  const { NL, NE } = a;
  return {
    alt: `Expert route for token ${sel}: ${NL} layers by ${NE} experts.`,
    size: (w) => ({ width: Math.max(w, LEFT + RIGHT + 200), height: NL * ROW + AXIS }),
    draw(ctx, W) {
      const pw = W - LEFT - RIGHT;
      const xOf = (e: number) => LEFT + ((e + 0.5) / NE) * pw;
      ctx.fillStyle = C.panel;
      ctx.fillRect(LEFT, 0, pw, NL * ROW);
      // orientation grid every 1/8 of the expert range
      ctx.fillStyle = 'rgba(255,255,255,0.04)';
      for (let k = 1; k < 8; k++) ctx.fillRect(Math.round(LEFT + (k / 8) * pw), 0, 1, NL * ROW);
      ctx.textBaseline = 'middle';

      type Dot = { x: number; r: number; e: number; w: number; prev: boolean };
      const dots: Dot[][] = [];
      for (let L = 0; L < NL; L++) {
        const y = L * ROW + ROW / 2;
        const r = a.rec(L, sel);
        const p = sel > 0 ? a.rec(L, sel - 1) : undefined;
        const fa = a.isFullAttention(L);
        ctx.font = font(11);
        ctx.fillStyle = fa ? C.dec : C.dim;
        ctx.fillText((fa ? '\u25c6 ' : '  ') + 'L' + L, 4, y);
        const row: Dot[] = [];
        dots.push(row);
        if (!r) continue;
        if (r.shared_gate != null) {
          ctx.fillStyle = C.line;
          ctx.fillRect(LEFT - 20, y - 4, 16, 8);
          ctx.fillStyle = C.pre;
          ctx.fillRect(LEFT - 20, y - 4, 16 * Math.max(0, Math.min(1, r.shared_gate)), 8);
        }
        if (p?.experts) {
          ctx.fillStyle = C.prev;
          p.experts.forEach((e, k) => {
            const x = xOf(e), rad = radius(p.weights?.[k]);
            ctx.beginPath();
            ctx.arc(x, y, rad, 0, 7);
            ctx.fill();
            row.push({ x, r: rad, e, w: p.weights?.[k] ?? NaN, prev: true });
          });
        }
        if (r.experts) {
          ctx.fillStyle = C.acc;
          r.experts.forEach((e, k) => {
            const x = xOf(e), rad = radius(r.weights?.[k]);
            ctx.beginPath();
            ctx.arc(x, y, rad, 0, 7);
            ctx.fill();
            row.push({ x, r: rad, e, w: r.weights?.[k] ?? NaN, prev: false });
          });
        }
        if (r.lens?.length) {
          const [t, pr] = r.lens[0];
          ctx.font = font(11);
          ctx.fillStyle = r.next_rank === 0 ? C.acc : C.fg;
          ctx.fillText(`${clip(vis(t), 12)} ${pct(pr, 0)}`, LEFT + pw + 8, y);
        }
      }
      if (!dots.some((row) => row.some((d) => !d.prev))) {
        const msg = sel < a.T && a.viz.tokens[sel]?.phase === 'prefill'
          ? 'No routing captured for this prefill token. Select a later token.'
          : 'No routing captured for this token.';
        ctx.font = font(12);
        ctx.textAlign = 'center';
        ctx.fillStyle = 'rgba(15,17,21,0.85)';
        const tw = ctx.measureText(msg).width;
        ctx.fillRect(LEFT + pw / 2 - tw / 2 - 12, (NL * ROW) / 2 - 16, tw + 24, 32);
        ctx.fillStyle = C.fg;
        ctx.fillText(msg, LEFT + pw / 2, (NL * ROW) / 2);
        ctx.textAlign = 'left';
      }
      ctx.textBaseline = 'alphabetic';
      ctx.font = font(10);
      ctx.fillStyle = C.dim;
      ctx.textAlign = 'center';
      for (const e of [0, NE >> 2, NE >> 1, (3 * NE) >> 2, NE - 1]) ctx.fillText(String(e), xOf(e), NL * ROW + 14);
      ctx.textAlign = 'left';
      ctx.fillText('expert id \u2192', LEFT + pw + 8, NL * ROW + 14);

      return (x, y): Hit | null => {
        const L = Math.floor(y / ROW);
        if (L < 0 || L >= NL) return null;
        const title = layerName(a, L);
        if (x < LEFT || x > LEFT + pw) {
          return { tip: { title, lines: recordLines(a, L, sel) } };
        }
        const cy = L * ROW + ROW / 2;
        let best: Dot | null = null;
        let bd = Infinity;
        for (const d of dots[L]) {
          const dd = Math.hypot(d.x - x, cy - y);
          // prefer the selected token's dot when both overlap
          const score = dd - (d.prev ? 0 : 0.5);
          if (dd <= d.r + 2 && score < bd) {
            bd = score;
            best = d;
          }
        }
        if (!best) return null;
        const who = best.prev ? `previous token #${sel - 1}` : `token #${sel}`;
        return {
          tip: {
            title: `L${L} expert ${best.e}`,
            lines: [
              { text: `${who} \u00b7 weight ${Number.isFinite(best.w) ? best.w.toFixed(3) : 'n/a'}`, tone: best.prev ? 'dim' : 'acc' },
              ...expertLines(a, L, best.e),
            ],
          },
        };
      };
    },
  };
}
