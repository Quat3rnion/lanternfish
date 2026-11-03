<script lang="ts">
  // Modal: start a trace run and watch the run list.
  import { api } from '../lib/api';
  import { bytes } from '../lib/format';
  import type { ModelFile, Run } from '../lib/types';

  let {
    open = $bindable(false),
    runs,
    runsError,
    traceNames,
    onstarted,
    onopen,
  }: {
    open: boolean;
    runs: Run[];
    runsError: string | null;
    traceNames: string[];
    onstarted: () => void;
    onopen: (name: string) => void;
  } = $props();

  let dialog: HTMLDialogElement;
  let models = $state<ModelFile[] | null>(null);
  let modelsError = $state<string | null>(null);
  let model = $state('');
  let name = $state('');
  let prompt = $state('');
  let n = $state(32);
  let raw = $state(false);
  let submitting = $state(false);
  let submitError = $state<string | null>(null);
  let notice = $state<string | null>(null);

  const NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;
  const nameTaken = $derived(traceNames.includes(name.trim()) || runs.some((r) => r.name === name.trim() && r.state !== 'failed'));
  const nameError = $derived(
    !name.trim() ? null : !NAME_RE.test(name.trim()) ? 'Use 1–64 characters: letters, digits, “.”, “_” and “-”.' : nameTaken ? 'A trace with this name exists.' : null,
  );
  const canSubmit = $derived(!!model && !!name.trim() && !nameError && !!prompt.trim() && n >= 1 && n <= 4096 && !submitting);

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
      if (!models) loadModels();
      if (!name) name = suggestName();
    } else if (!open && dialog.open) dialog.close();
  });

  function suggestName() {
    const d = new Date();
    const p = (x: number) => String(x).padStart(2, '0');
    return `trace-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  }

  async function loadModels() {
    modelsError = null;
    try {
      models = await api.models();
      if (!model && models.length) model = models[0].path;
    } catch (e) {
      modelsError = e instanceof Error ? e.message : String(e);
    }
  }

  async function submit(ev: SubmitEvent) {
    ev.preventDefault();
    if (!canSubmit) return;
    submitting = true;
    submitError = null;
    notice = null;
    try {
      const { id } = await api.startRun({ model, name: name.trim(), prompt, n, raw });
      notice = `Run ${id} started for “${name.trim()}”.`;
      name = suggestName() === name.trim() ? `${name.trim()}-2` : suggestName();
      onstarted();
    } catch (e) {
      submitError = e instanceof Error ? e.message : String(e);
    } finally {
      submitting = false;
    }
  }

  function modelName(path: string) {
    return models?.find((m) => m.path === path)?.name ?? path.split('/').pop() ?? path;
  }
</script>

<dialog bind:this={dialog} onclose={() => (open = false)} aria-labelledby="nt-title">
  <div class="head">
    <h2 id="nt-title">New trace</h2>
    <button type="button" class="close" aria-label="Close" onclick={() => (open = false)}>✕</button>
  </div>

  <form onsubmit={submit} novalidate>
    <label class="field">
      <span>Model</span>
      {#if modelsError}
        <span class="error" role="alert">Could not load models: {modelsError} <button type="button" onclick={loadModels}>Retry</button></span>
      {:else if !models}
        <span class="dim">Loading models…</span>
      {:else if !models.length}
        <span class="dim">The server found no GGUF files.</span>
      {:else}
        <select bind:value={model} required>
          {#each models as m (m.path)}
            <option value={m.path} title={m.path}>{m.name} · {bytes(m.size_bytes)}</option>
          {/each}
        </select>
        <span class="hint mono" title={model}>{model}</span>
      {/if}
    </label>

    <div class="row">
      <label class="field grow">
        <span>Name</span>
        <input
          type="text"
          bind:value={name}
          required
          spellcheck="false"
          autocomplete="off"
          aria-invalid={!!nameError}
          aria-describedby="nt-name-err"
        />
        <span id="nt-name-err" class="error small" aria-live="polite">{nameError ?? ''}</span>
      </label>
      <label class="field">
        <span>Tokens to decode</span>
        <input type="number" bind:value={n} min="1" max="4096" step="1" required />
      </label>
    </div>

    <label class="field">
      <span>Prompt</span>
      <textarea bind:value={prompt} rows="6" required spellcheck="false" placeholder="Write a Python function that returns the n-th Fibonacci number."></textarea>
    </label>

    <label class="check">
      <input type="checkbox" bind:checked={raw} />
      <span>Raw prompt <span class="dim">(send the text as-is, without the model's chat template)</span></span>
    </label>

    <div class="actions">
      <button type="submit" class="primary" disabled={!canSubmit} aria-busy={submitting}>
        {submitting ? 'Starting…' : 'Start trace'}
      </button>
      <span class="status" aria-live="polite">
        {#if submitError}<span class="error">{submitError}</span>{:else if notice}<span class="ok">{notice}</span>{/if}
      </span>
    </div>
  </form>

  <h3>Runs</h3>
  {#if runsError}
    <p class="error">Could not load runs: {runsError}</p>
  {/if}
  {#if !runs.length}
    <p class="dim">No runs yet.</p>
  {:else}
    <ul class="runs">
      {#each runs as r (r.id)}
        <li class="run state-{r.state}">
          <div class="run-head">
            <b class="mono">{r.name}</b>
            <span class="badge">{r.state}</span>
            <span class="dim small">{modelName(r.model)}</span>
            {#if r.state === 'done'}
              <button type="button" class="primary small-btn" onclick={() => onopen(r.name)}>Open trace</button>
            {/if}
          </div>
          <div class="progress-row">
            <div
              class="progress"
              role="progressbar"
              aria-label="Progress of {r.name}"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(r.progress * 100)}
            >
              <span style:width="{Math.max(0, Math.min(1, r.progress)) * 100}%"></span>
            </div>
            <span class="small num">{Math.round(r.progress * 100)}%</span>
            <span class="small dim">{r.phase}</span>
          </div>
          {#if r.error}<p class="error small">{r.error}</p>{/if}
          {#if r.log_tail}<pre class="log">{r.log_tail}</pre>{/if}
        </li>
      {/each}
    </ul>
  {/if}
</dialog>

<style>
  dialog {
    width: min(720px, calc(100vw - 24px));
    max-height: calc(100vh - 40px);
    padding: 0 20px 18px;
    background: var(--panel);
    color: var(--fg);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
  }
  dialog::backdrop {
    background: rgba(5, 6, 9, 0.7);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: var(--panel);
    padding: 14px 0 8px;
    z-index: 1;
  }
  h2 {
    margin: 0;
    font-size: 15px;
  }
  h3 {
    margin: 20px 0 8px;
    font-size: 13px;
    border-top: 1px solid var(--line);
    padding-top: 14px;
  }
  .close {
    width: 28px;
    height: 28px;
    padding: 0;
  }
  form {
    display: grid;
    gap: 12px;
  }
  .field {
    display: grid;
    gap: 4px;
  }
  .field > span:first-child {
    color: var(--dim);
    font-size: 12px;
  }
  .row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }
  .grow {
    flex: 1 1 260px;
  }
  .row input[type='number'] {
    width: 130px;
  }
  .hint {
    font-size: 11px;
    color: var(--dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  textarea {
    resize: vertical;
    min-height: 90px;
    font-family: var(--mono);
    font-size: 12.5px;
    line-height: 1.5;
  }
  .check {
    display: flex;
    gap: 8px;
    align-items: baseline;
  }
  .actions {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }
  .error {
    color: var(--warn);
  }
  .ok {
    color: var(--acc);
  }
  .small {
    font-size: 11.5px;
  }
  .num {
    font-variant-numeric: tabular-nums;
    min-width: 3.5ch;
    text-align: right;
  }
  .runs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 10px;
  }
  .run {
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 8px 10px;
    background: var(--bg);
  }
  .run-head {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .run-head .small-btn {
    margin-left: auto;
  }
  .small-btn {
    padding: 2px 10px;
    font-size: 12px;
  }
  .badge {
    font-size: 11px;
    padding: 0 6px;
    border-radius: 3px;
    border: 1px solid var(--line);
    color: var(--dim);
  }
  .state-running .badge { color: var(--pre); border-color: var(--pre); }
  .state-done .badge { color: var(--acc); border-color: var(--acc); }
  .state-failed .badge { color: var(--warn); border-color: var(--warn); }
  .progress-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
  }
  .progress {
    flex: 1;
    height: 6px;
    background: var(--line);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress span {
    display: block;
    height: 100%;
    background: var(--pre);
    transition: width 0.4s ease;
  }
  .state-done .progress span { background: var(--acc); }
  .state-failed .progress span { background: var(--warn); }
  .log {
    margin: 6px 0 0;
    padding: 6px 8px;
    background: #0a0c10;
    border-radius: 3px;
    font-size: 11px;
    line-height: 1.45;
    color: #b4bac6;
    max-height: 7.5em;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  @media (prefers-reduced-motion: reduce) {
    .progress span { transition: none; }
  }
</style>
