<script lang="ts">
  // devicePixelRatio-correct canvas that redraws when its painter or its
  // container width changes, with hover tooltips and click-to-select.
  import { onMount } from 'svelte';
  import type { HitTest, Painter } from '../lib/views/painter';
  import { hideTip, showTip } from '../lib/tooltip.svelte';

  let { painter, onpick }: { painter: Painter; onpick?: (token: number) => void } = $props();

  let wrap: HTMLDivElement;
  let canvas: HTMLCanvasElement;
  let width = $state(0);
  let dpr = $state(typeof devicePixelRatio === 'number' ? devicePixelRatio : 1);
  let hit: HitTest | null = null;
  let clickable = $state(false);

  onMount(() => {
    const ro = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0].contentRect.width);
      if (w !== width) width = w;
    });
    ro.observe(wrap);
    // devicePixelRatio changes with browser zoom or when the window moves to another screen
    let mq: MediaQueryList | null = null;
    const watchDpr = () => {
      mq?.removeEventListener('change', watchDpr);
      dpr = devicePixelRatio || 1;
      mq = matchMedia(`(resolution: ${dpr}dppx)`);
      mq.addEventListener('change', watchDpr);
    };
    watchDpr();
    return () => {
      ro.disconnect();
      mq?.removeEventListener('change', watchDpr);
    };
  });

  $effect(() => {
    const p = painter;
    const w = width;
    const r = dpr;
    if (!w || !canvas) return;
    const { width: cw, height: ch } = p.size(w);
    canvas.width = Math.round(cw * r);
    canvas.height = Math.round(ch * r);
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    hit = p.draw(ctx, cw, ch);
  });

  function at(ev: MouseEvent) {
    const b = canvas.getBoundingClientRect();
    return hit?.(ev.clientX - b.left, ev.clientY - b.top) ?? null;
  }

  function move(ev: MouseEvent) {
    const h = at(ev);
    clickable = h?.token != null && !!onpick;
    if (h) showTip(h.tip, ev.clientX, ev.clientY);
    else hideTip();
  }

  function click(ev: MouseEvent) {
    const h = at(ev);
    if (h?.token != null) onpick?.(h.token);
  }
</script>

<div class="wrap" bind:this={wrap}>
  <canvas
    bind:this={canvas}
    class:clickable
    onmousemove={move}
    onmouseleave={hideTip}
    onclick={click}>{painter.alt}</canvas
  >
</div>

<style>
  .wrap {
    width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
  }
  canvas {
    display: block;
  }
  canvas.clickable {
    cursor: pointer;
  }
</style>
