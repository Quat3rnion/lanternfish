import type { Tip } from './views/painter';

/** One shared tooltip for all canvases. */
export const tooltip = $state<{ tip: Tip | null; x: number; y: number }>({ tip: null, x: 0, y: 0 });

export function showTip(tip: Tip, x: number, y: number) {
  tooltip.tip = tip;
  tooltip.x = x;
  tooltip.y = y;
}

export function hideTip() {
  tooltip.tip = null;
}
