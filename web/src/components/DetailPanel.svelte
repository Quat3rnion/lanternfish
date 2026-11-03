<script lang="ts">
  import type { Analysis } from '../lib/analysis';
  import { int, pct } from '../lib/format';
  import Piece from './Piece.svelte';

  let { a, sel }: { a: Analysis; sel: number } = $props();

  const tok = $derived(a.viz.tokens[sel]);
  const stepIdx = $derived(a.stepEnd[sel] ?? -1);
  const cands = $derived(stepIdx >= 0 ? (a.viz.next_candidates[stepIdx] ?? []) : []);
  const emitted = $derived(a.viz.tokens[sel + 1]?.id);
  const rows = $derived(
    Array.from({ length: a.NL }, (_, L) => ({ L, r: a.rec(L, sel) })).filter((x) => x.r),
  );
  const hasGate = $derived(rows.some((x) => x.r?.shared_gate != null));
</script>

<section class="detail" aria-label="Selected token details">
  {#if tok}
    <h2>
      <span class="mono piece"><Piece piece={tok.piece} /></span>
      <span class="meta">#{sel} · <span class="ph-{tok.phase}">{tok.phase}</span> · id {tok.id}</span>
    </h2>
    {#if a.stepOf[sel] >= 0}
      <p class="dim small">Step {a.stepOf[sel] + 1} of {a.viz.steps.length}{stepIdx >= 0 ? ' (last token of the step)' : ''}</p>
    {/if}

    {#if cands.length}
      <h3>Next-token distribution after this token</h3>
      <table class="cands">
        <thead class="sr-only"><tr><th>Token</th><th>Probability</th></tr></thead>
        <tbody>
          {#each cands as c (c.id)}
            <tr class:emitted={c.id === emitted}>
              <td class="mono"><Piece piece={c.piece} max={18} />{#if c.id === emitted}<span class="tag">emitted</span>{/if}</td>
              <td class="p">
                <span class="bar" style:width="{Math.max(0.5, c.p * 100)}%"></span>
                <span class="num">{pct(c.p, 1)}</span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else if stepIdx < 0}
      <p class="dim small">This token is inside a prefill batch, so the model made no next-token prediction for it.</p>
    {/if}

    <h3>Per layer</h3>
    <p class="dim small">Logit-lens top 3 · rank of the emitted next token{hasGate ? ' · shared-expert gate' : ''}</p>
    <table class="layers">
      <thead>
        <tr>
          <th scope="col">Layer</th>
          <th scope="col">Lens top 3</th>
          <th scope="col" class="r">Rank</th>
          {#if hasGate}<th scope="col" class="r">Gate</th>{/if}
        </tr>
      </thead>
      <tbody>
        {#each rows as { L, r } (L)}
          {@const fa = a.isFullAttention(L)}
          <tr>
            <th scope="row" class:fa title={a.viz.layers[L].kind}>{fa ? '\u25c6' : ''}L{L}</th>
            <td class="mono lens">
              {#each r!.lens.slice(0, 3) as [t, p], k (k)}
                <span class="chip"><Piece piece={t} max={12} /><span class="dim pp">{pct(p)}</span></span>
              {/each}
            </td>
            <td class="r num" class:top={r!.next_rank === 0}>{r!.next_rank != null ? int(r!.next_rank) : '\u2013'}</td>
            {#if hasGate}<td class="r num">{r!.shared_gate != null ? r!.shared_gate.toFixed(3) : ''}</td>{/if}
          </tr>
        {/each}
      </tbody>
    </table>
  {:else}
    <p class="dim">Select a token.</p>
  {/if}
</section>

<style>
  .detail {
    font-size: 12px;
  }
  h2 {
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 2px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: baseline;
  }
  .piece {
    color: var(--acc);
    word-break: break-all;
  }
  .meta {
    font-weight: 400;
    font-size: 12px;
    color: var(--dim);
  }
  .ph-prefill { color: var(--pre); }
  .ph-decode { color: var(--dec); }
  h3 {
    font-size: 12px;
    font-weight: 600;
    margin: 14px 0 4px;
    color: var(--fg);
  }
  .small {
    margin: 2px 0 6px;
    font-size: 12px;
  }
  table {
    border-collapse: collapse;
    width: 100%;
  }
  td, th {
    padding: 2px 4px;
    border-bottom: 1px solid var(--row-line);
    vertical-align: top;
    text-align: left;
    font-weight: 400;
  }
  thead th {
    color: var(--dim);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 1;
  }
  .r { text-align: right; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  th[scope='row'] { color: var(--dim); white-space: nowrap; }
  th.fa { color: var(--dec); }
  .lens { line-height: 1.5; }
  .chip { display: inline-block; margin-right: 8px; white-space: nowrap; }
  .pp { margin-left: 0.5ch; }
  .top { color: var(--acc); font-weight: 600; }
  .cands td { border-bottom-color: transparent; }
  .cands .p {
    position: relative;
    width: 45%;
    text-align: right;
  }
  .bar {
    position: absolute;
    left: 0;
    top: 3px;
    bottom: 3px;
    background: var(--bar);
    border-radius: 1px;
  }
  .p .num { position: relative; }
  .emitted td { color: var(--acc); }
  .tag {
    margin-left: 6px;
    font-family: var(--sans);
    font-size: 10px;
    padding: 0 4px;
    border: 1px solid var(--acc);
    border-radius: 3px;
  }
</style>
