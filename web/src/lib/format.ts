// Text helpers shared by the DOM views and the canvas painters.

/** Visible form of a token piece: newline -> ↵, space -> ·, tab -> →, empty -> ∅. */
export function vis(s: string): string {
  if (!s) return '\u2205';
  return s.replace(/\r?\n/g, '\u21b5').replace(/ /g, '\u00b7').replace(/\t/g, '\u2192');
}

/** Truncate to n code points (not UTF-16 units) and add an ellipsis. */
export function clip(s: string, n: number): string {
  const cps = Array.from(s);
  return cps.length <= n ? s : cps.slice(0, n - 1).join('') + '\u2026';
}

export type Seg = { kind: 'text' | 'space' | 'nl' | 'tab'; s: string };

/** Split a piece into segments so whitespace can be styled separately. */
export function segments(piece: string): Seg[] {
  const out: Seg[] = [];
  let buf = '';
  const flush = () => {
    if (buf) out.push({ kind: 'text', s: buf });
    buf = '';
  };
  for (const ch of piece) {
    if (ch === ' ') {
      flush();
      out.push({ kind: 'space', s: '\u00b7' });
    } else if (ch === '\n') {
      flush();
      out.push({ kind: 'nl', s: '\u21b5' });
    } else if (ch === '\t') {
      flush();
      out.push({ kind: 'tab', s: '\u2192' });
    } else if (ch !== '\r') {
      buf += ch;
    }
  }
  flush();
  if (!out.length) out.push({ kind: 'space', s: '\u2205' });
  return out;
}

export function pct(p: number, digits?: number): string {
  const d = digits ?? (p < 0.1 ? 1 : 0);
  return `${(p * 100).toFixed(d)}%`;
}

const nf = new Intl.NumberFormat('en-US');
export function int(n: number): string {
  return nf.format(n);
}

export function bytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${Math.round(n / 1e3)} kB`;
}

export function num(v: number | null | undefined, digits = 3): string {
  return v == null || Number.isNaN(v) ? '\u2013' : v.toFixed(digits);
}
