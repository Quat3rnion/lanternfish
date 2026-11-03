<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { analyse, comparable, type Analysis } from './lib/analysis';
  import { api } from './lib/api';
  import type { CompareMode, Run, TraceSummary, ViewId } from './lib/types';
  import { readUrl, writeUrl } from './lib/url';
  import DetailPanel from './components/DetailPanel.svelte';
  import NewTraceDialog from './components/NewTraceDialog.svelte';
  import TokenStrip from './components/TokenStrip.svelte';
  import Tooltip from './components/Tooltip.svelte';
  import ViewPane from './components/ViewPane.svelte';

  const initial = readUrl();

  let traces = $state<TraceSummary[] | null>(null);
  let tracesError = $state<string | null>(null);
  let traceName = $state<string | null>(initial.t);
  let a = $state.raw<Analysis | null>(null);
  let loadError = $state<string | null>(null);
  let loading = $state(false);
  let sel = $state(0);
  let view = $state<ViewId>(initial.view ?? 'route');
  let cmpName = $state<string | null>(initial.cmp);
  let b = $state.raw<Analysis | null>(null);
  let cmpError = $state<string | null>(null);
  let cmode = $state<CompareMode>(initial.cmode ?? 'side');
  let playing = $state(false);
  let dialogOpen = $state(false);
  let runs = $state<Run[]>([]);
  let runsError = $state<string | null>(null);
  let readyRun = $state<string | null>(null);

  // first selection after a load comes from the URL, later loads start at the first decode token
  let pendingTok: number | null = initial.tok;

  const moe = $derived(!!a && a.NE > 0);
  const VIEWS: { id: ViewId; label: string; moe: boolean }[] = [
    { id: 'route', label: 'Expert route', moe: true },
    { id: 'lens', label: 'Logit lens', moe: false },
    { id: 'write', label: 'Write energy', moe: false },
    { id: 'usage', label: 'Expert usage', moe: true },
  ];
  const views = $derived(VIEWS.filter((v) => !v.moe || moe));
  const effView = $derived<ViewId>(!a ? view : !moe && (view === 'route' || view === 'usage') ? 'lens' : view);
  const current = $derived(traces?.find((t) => t.name === traceName) ?? null);
  const candidates = $derived(
    a && traces ? traces.filter((t) => t.name !== traceName && comparable(t.model, { arch: a!.viz.model.arch, n_layers: a!.NL, n_experts: a!.NE })) : [],
  );
  const compareView = $derived(effView === 'lens' || (effView === 'usage' && moe));
  const activeRuns = $derived(runs.some((r) => r.state === 'queued' || r.state === 'running'));

  async function loadTraces() {
    try {
      traces = await api.traces();
      tracesError = null;
      if (!traceName && traces.length) traceName = traces[0].name;
    } catch (e) {
      tracesError = e instanceof Error ? e.message : String(e);
      traces ??= [];
    }
  }

  // load the main trace
  $effect(() => {
    const name = traceName;
    if (!name) return;
    let stale = false;
    untrack(() => {
      loading = true;
      loadError = null;
      stopPlay();
    });
    api
      .trace(name)
      .then((viz) => {
        if (stale) return;
        const an = analyse(viz);
        const want = pendingTok;
        pendingTok = null;
        sel = want != null && want < an.T ? want : Math.max(0, an.firstDecode);
        a = an;
      })
      .catch((e) => {
        if (stale) return;
        a = null;
        loadError = e instanceof Error ? e.message : String(e);
      })
      .finally(() => {
        if (!stale) loading = false;
      });
    return () => {
      stale = true;
    };
  });

  // load the comparison trace
  $effect(() => {
    const name = cmpName;
    if (!name) {
      b = null;
      cmpError = null;
      return;
    }
    let stale = false;
    api
      .trace(name)
      .then((viz) => {
        if (stale) return;
        b = analyse(viz);
        cmpError = null;
      })
      .catch((e) => {
        if (!stale) cmpError = e instanceof Error ? e.message : String(e);
      });
    return () => {
      stale = true;
    };
  });

  // drop a comparison that no longer matches the main trace
  $effect(() => {
    if (a && b && !comparable({ arch: a.viz.model.arch, n_layers: a.NL, n_experts: a.NE }, { arch: b.viz.model.arch, n_layers: b.NL, n_experts: b.NE })) {
      cmpName = null;
    }
  });

  $effect(() => {
    writeUrl({ t: traceName, tok: a ? sel : null, view: effView, cmp: cmpName, cmode });
  });

  function select(i: number) {
    if (!a) return;
    sel = Math.max(0, Math.min(a.T - 1, i));
  }

  let timer: ReturnType<typeof setInterval> | null = null;
  function stopPlay() {
    if (timer) clearInterval(timer);
    timer = null;
    playing = false;
  }
  function togglePlay() {
    if (!a) return;
    if (playing) return stopPlay();
    const fd = a.firstDecode;
    if (fd < 0) return;
    if (sel < fd || sel >= a.T - 1) sel = fd;
    playing = true;
    timer = setInterval(() => {
      if (!a || sel >= a.T - 1) return stopPlay();
      sel++;
    }, 450);
  }

  function onKey(e: KeyboardEvent) {
    if (!a || dialogOpen || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.closest('input, textarea, select, [contenteditable]') || (t.tagName === 'BUTTON' && t.closest('[role="tablist"]')))) return;
    if (e.key === 'ArrowRight') select(sel + 1);
    else if (e.key === 'ArrowLeft') select(sel - 1);
    else if (e.key === 'Home') select(0);
    else if (e.key === 'End') select(a.T - 1);
    else return;
    e.preventDefault();
  }

  function onTabKey(e: KeyboardEvent) {
    const i = views.findIndex((v) => v.id === effView);
    let j = -1;
    if (e.key === 'ArrowRight') j = (i + 1) % views.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + views.length) % views.length;
    else return;
    e.preventDefault();
    view = views[j].id;
    document.getElementById(`tab-${view}`)?.focus();
  }

  // run polling: every 2 s while the dialog is open or a run is in progress
  let knownDone = new Set<string>();
  let runsSeeded = false;
  async function pollRuns() {
    try {
      const list = await api.runs();
      runsError = null;
      const newlyDone = list.filter((r) => r.state === 'done' && !knownDone.has(r.id));
      for (const r of newlyDone) knownDone.add(r.id);
      runs = list;
      if (runsSeeded && newlyDone.length) {
        await loadTraces();
        readyRun = newlyDone[newlyDone.length - 1].name;
      }
      runsSeeded = true;
    } catch (e) {
      runsError = e instanceof Error ? e.message : String(e);
    }
  }

  $effect(() => {
    if (!dialogOpen && !activeRuns) return;
    const id = setInterval(pollRuns, 2000);
    return () => clearInterval(id);
  });

  function openTrace(name: string) {
    dialogOpen = false;
    readyRun = null;
    if (name === traceName) return;
    pendingTok = null;
    cmpName = null;
    traceName = name;
  }

  function openDialog() {
    dialogOpen = true;
    pollRuns();
  }

  onMount(() => {
    loadTraces();
    pollRuns();
    return stopPlay;
  });

  function created(s: string) {
    return s ? s.replace('T', ' ').slice(0, 16) : '';
  }
</script>

<svelte:window onkeydown={onKey} />

<header>
  <h1>lanternfish</h1>
  <label class="picker">
    <span class="sr-only">Trace</span>
    <select
      aria-label="Trace"
      value={traceName ?? ''}
      onchange={(e) => {
        pendingTok = null;
        traceName = e.currentTarget.value;
      }}
      disabled={!traces?.length}
    >
      {#if !traces}<option value="">Loading…</option>{/if}
      {#each traces ?? [] as t (t.name)}
        <option value={t.name}>{t.name} · {t.model.name} · {t.tokens} tok · {created(t.created)}</option>
      {/each}
      {#if traceName && traces && !current}<option value={traceName}>{traceName}</option>{/if}
    </select>
  </label>

  <div class="tabs" role="tablist" aria-label="View" tabindex="-1" onkeydown={onTabKey}>
    {#each views as v (v.id)}
      <button
        id="tab-{v.id}"
        role="tab"
        aria-selected={effView === v.id}
        aria-controls="view-panel"
        tabindex={effView === v.id ? 0 : -1}
        class:on={effView === v.id}
        onclick={() => (view = v.id)}>{v.label}</button
      >
    {/each}
  </div>

  <button onclick={togglePlay} disabled={!a || a.firstDecode < 0} aria-pressed={playing} aria-label={playing ? 'Pause decode playback' : 'Play decode tokens'}>
    {playing ? '❚❚ pause' : '▶ play decode'}
  </button>

  <span class="spacer"></span>
  <button class="primary" onclick={openDialog}>+ New trace{#if activeRuns}<span class="pulse" aria-label="runs in progress"></span>{/if}</button>
</header>

<div class="subbar">
  {#if a}
    <span class="meta">
      <b>{a.viz.model.name}</b>
      <span class="dim">({a.viz.model.arch}, {a.NL} layers{a.NE ? `, ${a.NE} experts top-${a.viz.top_k}` : ', dense'}{a.hybrid ? ', hybrid attention' : ''})</span>
      · {a.T} tokens
      {#if a.viz.lens_last_layer_agreement != null}
        · <span title="Share of tokens where the logit lens at the last layer agrees with the emitted next token">last-layer lens agreement {(a.viz.lens_last_layer_agreement * 100).toFixed(1)}%</span>
      {/if}
    </span>
  {/if}
  <span class="compare">
    <label>
      <span class="dim">Compare with</span>
      <select
        aria-label="Compare with trace"
        value={cmpName ?? ''}
        onchange={(e) => (cmpName = e.currentTarget.value || null)}
        disabled={!a || (!candidates.length && !cmpName)}
      >
        <option value="">{candidates.length ? 'none' : 'no trace of the same model'}</option>
        {#each candidates as t (t.name)}
          <option value={t.name}>{t.name} · {t.tokens} tok</option>
        {/each}
      </select>
    </label>
    {#if cmpName}
      <span class="seg" role="radiogroup" aria-label="Compare layout">
        <button role="radio" aria-checked={cmode === 'side'} class:on={cmode === 'side'} onclick={() => (cmode = 'side')}>Side by side</button>
        <button role="radio" aria-checked={cmode === 'diff'} class:on={cmode === 'diff'} onclick={() => (cmode = 'diff')}>Difference</button>
      </span>
    {/if}
  </span>
  <span class="keys" role="group" aria-label="Step token">
    <button class="step" onclick={() => select(sel - 1)} disabled={!a || sel <= 0} aria-label="Previous token" title="Previous token (←)">←</button>
    <button class="step" onclick={() => select(sel + 1)} disabled={!a || sel >= a.T - 1} aria-label="Next token" title="Next token (→)">→</button>
    <span class="dim">step token</span>
  </span>
</div>

{#if readyRun && !dialogOpen}
  <div class="toast" role="status">
    Trace <b class="mono">{readyRun}</b> is ready.
    <button class="primary" onclick={() => openTrace(readyRun!)}>Open</button>
    <button onclick={() => (readyRun = null)} aria-label="Dismiss">Dismiss</button>
  </div>
{/if}

{#if a}
  <TokenStrip tokens={a.viz.tokens} {sel} onselect={select} />
{/if}

<main>
  <div id="view-panel" class="left" role="tabpanel" aria-labelledby="tab-{effView}" aria-busy={loading}>
    {#if tracesError && !traces?.length}
      <p class="error" role="alert">Could not load the trace list: {tracesError}</p>
    {:else if traces && !traces.length}
      <div class="empty">
        <p>There are no traces yet.</p>
        <button class="primary" onclick={openDialog}>Create a trace</button>
      </div>
    {:else if loadError}
      <p class="error" role="alert">Could not load trace “{traceName}”: {loadError}</p>
    {:else if !a}
      <p class="dim">Loading trace…</p>
    {:else}
      {#if cmpName && !compareView}
        <p class="note">Compare mode shows the Logit lens{moe ? ' and Expert usage views' : ' view'}. This view shows only <b>{a.viz.name}</b>.</p>
      {/if}
      {#if cmpError}<p class="error" role="alert">Could not load “{cmpName}”: {cmpError}</p>{/if}
      <div class:fading={loading}>
        <ViewPane {a} b={compareView ? b : null} view={effView} {sel} mode={cmode} onpick={select} />
      </div>
    {/if}
  </div>
  <aside class="right">
    {#if a}<DetailPanel {a} {sel} />{/if}
  </aside>
</main>

<Tooltip />
<NewTraceDialog
  bind:open={dialogOpen}
  {runs}
  {runsError}
  traceNames={(traces ?? []).map((t) => t.name)}
  onstarted={pollRuns}
  onopen={openTrace}
/>

<style>
  header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--line);
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }
  h1 {
    font-size: 15px;
    margin: 0 6px 0 0;
    font-weight: 650;
    letter-spacing: 0.01em;
  }
  .picker select {
    max-width: min(460px, 70vw);
  }
  .tabs {
    display: inline-flex;
    gap: 4px;
  }
  .spacer {
    flex: 1;
  }
  .pulse {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-left: 7px;
    border-radius: 50%;
    background: var(--dec);
    vertical-align: middle;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    50% { opacity: 0.25; }
  }
  .subbar {
    display: flex;
    gap: 10px 20px;
    align-items: center;
    flex-wrap: wrap;
    padding: 6px 16px;
    border-bottom: 1px solid var(--line);
    font-size: 12px;
  }
  .meta b {
    font-weight: 600;
  }
  .compare {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .compare label {
    display: inline-flex;
    gap: 6px;
    align-items: center;
  }
  .compare select {
    font-size: 12px;
    padding: 2px 6px;
    max-width: 280px;
  }
  .seg {
    display: inline-flex;
  }
  .seg button {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 0;
  }
  .seg button:first-child { border-radius: 4px 0 0 4px; }
  .seg button:last-child { border-radius: 0 4px 4px 0; margin-left: -1px; }
  .step {
    min-width: 28px;
    padding: 2px 8px;
    font-family: ui-monospace, monospace;
  }
  .step:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .keys {
    margin-left: auto;
    display: inline-flex;
    gap: 4px;
    align-items: center;
  }
  main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 380px;
  }
  .left {
    padding: 4px 16px 24px;
    min-width: 0;
  }
  .right {
    border-left: 1px solid var(--line);
    padding: 10px 14px 24px;
    min-width: 0;
  }
  .fading {
    opacity: 0.55;
    transition: opacity 0.15s;
  }
  .note {
    font-size: 12px;
    color: var(--dim);
    border-left: 2px solid var(--dec);
    padding-left: 8px;
  }
  .error {
    color: var(--warn);
  }
  .empty {
    padding: 40px 0;
    text-align: center;
    color: var(--dim);
  }
  .toast {
    position: fixed;
    right: 16px;
    bottom: 16px;
    z-index: 40;
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    background: var(--panel);
    border: 1px solid var(--acc);
    border-radius: 6px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  }
  @media (max-width: 980px) {
    main {
      grid-template-columns: minmax(0, 1fr);
    }
    .right {
      border-left: 0;
      border-top: 1px solid var(--line);
    }
    .keys {
      display: none;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .pulse { animation: none; }
    .fading { transition: none; }
  }
</style>
