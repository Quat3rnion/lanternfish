<script lang="ts">
  import { tooltip } from '../lib/tooltip.svelte';

  let el: HTMLDivElement | undefined = $state();
  let w = $state(0);
  let h = $state(0);

  const pos = $derived.by(() => {
    const vw = typeof innerWidth === 'number' ? innerWidth : 1200;
    const vh = typeof innerHeight === 'number' ? innerHeight : 800;
    let x = tooltip.x + 14;
    let y = tooltip.y + 16;
    if (x + w > vw - 8) x = Math.max(8, tooltip.x - w - 14);
    if (y + h > vh - 8) y = Math.max(8, tooltip.y - h - 12);
    return { x, y };
  });
</script>

{#if tooltip.tip}
  {@const t = tooltip.tip}
  <div
    class="tip"
    bind:this={el}
    bind:offsetWidth={w}
    bind:offsetHeight={h}
    style:left="{pos.x}px"
    style:top="{pos.y}px"
    role="tooltip"
  >
    <div class="title">
      {#if t.titleMono}<span class="mono piece">{t.titleMono}</span>{/if}
      {t.title}
    </div>
    {#each t.lines as line, k (k)}
      <div class="line tone-{line.tone ?? 'fg'}">
        {#if line.label}<span class="label">{line.label}</span>{/if}
        <span class:mono={line.mono}>{line.text}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
  .tip {
    position: fixed;
    z-index: 50;
    pointer-events: none;
    max-width: min(420px, calc(100vw - 16px));
    padding: 7px 9px;
    background: rgba(8, 9, 12, 0.94);
    border: 1px solid var(--line);
    border-radius: 4px;
    font-size: 12px;
    line-height: 1.45;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);
  }
  .title {
    font-weight: 600;
    margin-bottom: 3px;
  }
  .piece {
    font-weight: 600;
    color: var(--acc);
    margin-right: 4px;
  }
  .line {
    overflow-wrap: anywhere;
  }
  .label {
    color: var(--dim);
    margin-right: 6px;
  }
  .tone-dim { color: var(--dim); }
  .tone-acc { color: var(--acc); }
  .tone-pre { color: var(--pre); }
  .tone-dec { color: var(--dec); }
  .tone-warn { color: var(--warn); }
  .tone-a { color: var(--a); }
  .tone-b { color: var(--b); }
</style>
