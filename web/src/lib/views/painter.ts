// Contract between Canvas.svelte and the view painters.

export type Tone = 'dim' | 'acc' | 'pre' | 'dec' | 'warn' | 'a' | 'b';

export interface TipLine {
  label?: string;
  text: string;
  mono?: boolean;
  tone?: Tone;
}

export interface Tip {
  title: string;
  /** optional mono part of the title, e.g. a token piece */
  titleMono?: string;
  lines: TipLine[];
}

export interface Hit {
  tip: Tip;
  /** token index to select when the user clicks the hit */
  token?: number;
}

export type HitTest = (x: number, y: number) => Hit | null;

export interface Painter {
  /** CSS-pixel size of the canvas for a container of the given width */
  size(containerWidth: number): { width: number; height: number };
  /** draw in CSS pixels (the transform for devicePixelRatio is already set) */
  draw(ctx: CanvasRenderingContext2D, width: number, height: number): HitTest;
  /** short text alternative for screen readers */
  alt: string;
}

export const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace';

export function font(px: number): string {
  return `${px}px ${MONO}`;
}
