<script lang="ts">
  // Legend + canvas(es) for the active view, single or compared.
  import type { Analysis } from '../lib/analysis';
  import { DIV_CSS, SEQ_CSS } from '../lib/colors';
  import type { CompareMode, ViewId } from '../lib/types';
  import { lensDiffPainter, lensPainter, writePainter } from '../lib/views/grid';
  import type { Painter } from '../lib/views/painter';
  import { routePainter } from '../lib/views/route';
  import { usageDiffPainter, usagePainter } from '../lib/views/usage';
  import Canvas from './Canvas.svelte';
  import ColorScale from './ColorScale.svelte';

  let {
    a,
    b = null,
    view,
    sel,
    mode = 'side',
    onpick,
  }: {
    a: Analysis;
    b?: Analysis | null;
    view: ViewId;
    sel: number;
    mode?: CompareMode;
    onpick: (i: number) => void;
  } = $props();

  function single(x: Analysis, v: ViewId, s: number): Painter {
    if (v === 'route' && x.NE) return routePainter(x, s);
    if (v === 'usage' && x.NE) return usagePainter(x, s);
    if (v === 'write') return writePainter(x, s);
    return lensPainter(x, s);
  }

  const comparing = $derived(!!b && (view === 'lens' || (view === 'usage' && a.NE > 0)));
  const pa = $derived(single(a, view, sel));
  const pb = $derived(b && comparing && mode === 'side' ? single(b, view, Math.min(sel, b.T - 1)) : null);
  const pd = $derived(
    b && comparing && mode === 'diff' ? (view === 'usage' ? usageDiffPainter(a, b) : lensDiffPainter(a, b, sel)) : null,
  );
  const fa = $derived(a.hybrid);
  const pickB = (i: number) => onpick(Math.min(i, a.T - 1));
</script>

<div class="legend">
  {#if pd && b}
    {#if view === 'lens'}
      <p>
        Logit-lens value of A (<b class="a">{a.viz.name}</b>) minus B (<b class="b">{b.viz.name}</b>), tokens aligned by
        position (first {Math.min(a.T, b.T)}). Pink: A ranks its emitted next token higher at that layer. Cyan: B does.
        <span class="warn">Red ticks</span> mark positions where the two traces hold different tokens.
      </p>
    {:else}
      <p>
        Share of routed tokens that each expert receives, A (<b class="a">{a.viz.name}</b>) minus B (<b class="b">{b.viz.name}</b>).
        Shares remove the effect of different token counts. Pink: A uses the expert more. Cyan: B uses it more.
      </p>
    {/if}
    <ColorScale gradient={DIV_CSS} left="B −{(pd.scale * (view === 'usage' ? 100 : 1)).toFixed(view === 'usage' ? 1 : 2)}{view === 'usage' ? ' pt' : ''}" right="A +{(pd.scale * (view === 'usage' ? 100 : 1)).toFixed(view === 'usage' ? 1 : 2)}{view === 'usage' ? ' pt' : ''}" label="Difference A minus B" />
  {:else if view === 'route'}
    <p>
      Rows: layers 0 (top) to {a.NL - 1}. X: routed expert id 0–{a.NE - 1}. Dot area: router weight after top-{a.viz.top_k}
      renormalization. <span class="acc">●</span> selected token, <span class="prev">●</span> previous token. Left bar:
      shared-expert gate. Right text: logit-lens top token after the layer (green when it is the emitted next token).
      {#if fa}Full-attention layers are marked <span class="dec">◆</span>.{/if}
    </p>
  {:else if view === 'lens'}
    <p>
      Cell (token, layer): how early the model “knows” the next token. Color: 1 − log₁₀(rank + 1) / 4 of the emitted next token
      in the logit lens after that layer. Dark grey: no next token. Click a column to select a token.
      {#if fa}<span class="dec">◆</span> full-attention layer.{/if}
    </p>
    <ColorScale gradient={SEQ_CSS} left="rank ≥ 9999" right="rank 0" label="Next-token rank" />
  {:else if view === 'write'}
    <p>
      Cell (token, layer): RMS that the layer writes into the residual (attention or linear attention + feed-forward), divided by
      the residual RMS. Bright: the layer changes the token's state a lot. Click a column to select a token.
    </p>
    <ColorScale gradient={SEQ_CSS} left="0" right="≥ {a.writeScale.toFixed(2)} (p99)" label="Write over residual RMS" />
  {:else}
    <p>
      Rows: layers. X: expert id. Bar height: number of tokens routed to the expert (scaled per layer).
      <span class="pre">■ prefill</span> <span class="dec">■ decode</span>. <span class="acc">▬</span> experts of the selected
      token. Shows how concentrated the routing is for this prompt.
    </p>
  {/if}
</div>

{#if pd}
  <Canvas painter={pd} {onpick} />
{:else if pb && b}
  <div class="side">
    <figure>
      <figcaption><b class="a">A</b> {a.viz.name} · {a.T} tokens</figcaption>
      <Canvas painter={pa} {onpick} />
    </figure>
    <figure>
      <figcaption><b class="b">B</b> {b.viz.name} · {b.T} tokens</figcaption>
      <Canvas painter={pb} onpick={pickB} />
    </figure>
  </div>
{:else}
  <Canvas painter={pa} {onpick} />
{/if}

<style>
  .legend {
    color: var(--dim);
    font-size: 12px;
    margin: 6px 0 10px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 16px;
  }
  .legend p {
    margin: 0;
    flex: 1 1 520px;
    max-width: 96ch;
  }
  .acc { color: var(--acc); }
  .pre { color: var(--pre); }
  .dec { color: var(--dec); }
  .prev { color: #9ca2ae; }
  .warn { color: var(--warn); }
  .a { color: var(--a); }
  .b { color: var(--b); }
  .side {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 12px 20px;
  }
  figure {
    margin: 0;
    min-width: 0;
  }
  figcaption {
    font-size: 12px;
    color: var(--dim);
    margin-bottom: 4px;
  }
</style>
