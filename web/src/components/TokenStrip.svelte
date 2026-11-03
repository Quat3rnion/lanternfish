<script lang="ts">
  import type { Token } from '../lib/types';
  import Piece from './Piece.svelte';

  let { tokens, sel, onselect }: { tokens: Token[]; sel: number; onselect: (i: number) => void } = $props();

  let box: HTMLDivElement;

  $effect(() => {
    const i = sel;
    const el = box?.querySelector<HTMLElement>(`#tok-${i}`);
    if (!el || !box) return;
    // keep the selected token visible without moving the page
    const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
    if (r.top < b.top) box.scrollTop -= b.top - r.top + 4;
    else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom + 4;
  });
</script>

<div
  class="strip mono"
  bind:this={box}
  role="listbox"
  tabindex="0"
  aria-label="Tokens. Use the arrow keys to step through tokens."
  aria-activedescendant="tok-{sel}"
>
  {#each tokens as t (t.i)}<span
      id="tok-{t.i}"
      class="tok {t.phase}"
      class:sel={t.i === sel}
      role="option"
      aria-selected={t.i === sel}
      title="#{t.i} · {t.phase} · id {t.id}"
      onclick={() => onselect(t.i)}
      onkeydown={() => {}}
      tabindex="-1"><Piece piece={t.piece} breaks /></span>{/each}
</div>

<style>
  .strip {
    padding: 8px 16px 10px;
    max-height: 150px;
    overflow: auto;
    line-height: 1.75;
    font-size: 13px;
    border-bottom: 1px solid var(--line);
    word-break: break-all;
  }
  .strip:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .tok {
    cursor: pointer;
    padding: 1px 1px 0;
    margin-right: 1px;
    border-radius: 2px;
    border-bottom: 2px solid transparent;
  }
  .tok.prefill {
    border-bottom-color: var(--pre-line);
  }
  .tok.decode {
    border-bottom-color: var(--dec-line);
    color: #f3e2cc;
  }
  .tok:hover {
    background: var(--hover);
  }
  .tok.sel {
    background: var(--sel-bg);
    outline: 1px solid var(--acc);
  }
</style>
