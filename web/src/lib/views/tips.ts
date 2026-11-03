import type { Analysis } from '../analysis';
import { int, num, pct, vis } from '../format';
import type { TipLine } from './painter';

export function expertLines(a: Analysis, L: number, e: number): TipLine[] {
  const lab = a.label(L, e);
  if (!lab) return [{ text: 'no output-direction label for this expert', tone: 'dim' }];
  return [
    { label: 'writes toward', text: lab.pos.map(vis).join('  '), mono: true },
    { label: 'away from', text: lab.neg.map(vis).join('  '), mono: true },
    { label: 'rank-1 share', text: String(lab.rank1), tone: 'dim' },
  ];
}

export function recordLines(a: Analysis, L: number, i: number): TipLine[] {
  const r = a.rec(L, i);
  if (!r) return [{ text: 'not captured for this token', tone: 'dim' }];
  const lines: TipLine[] = [
    { label: 'lens', text: r.lens.map(([t, p]) => `${vis(t)} ${pct(p)}`).join('  '), mono: true },
  ];
  if (r.next_rank != null)
    lines.push({ label: 'next-token rank', text: int(r.next_rank), tone: r.next_rank === 0 ? 'acc' : undefined });
  lines.push({
    label: 'write',
    text: `attn ${num(r.write?.attn ?? null)}  ffn ${num(r.write?.ffn ?? null)}  resid ${num(r.resid_rms)}`,
  });
  if (r.shared_gate != null) lines.push({ label: 'shared gate', text: num(r.shared_gate) });
  return lines;
}

export function layerName(a: Analysis, L: number): string {
  const kind = a.viz.layers[L]?.kind ?? 'unknown';
  return `L${L} \u00b7 ${kind === 'linear' ? 'linear attention' : kind === 'attention' ? 'full attention' : 'unknown kind'}`;
}
