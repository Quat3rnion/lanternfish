<script lang="ts">
  // A token piece with visible whitespace: space ·, newline ↵, tab →.
  import { segments } from '../lib/format';

  let { piece, breaks = false, max }: { piece: string; breaks?: boolean; max?: number } = $props();

  const segs = $derived.by(() => {
    const all = segments(piece);
    if (max == null) return all;
    let n = 0;
    const out = [];
    for (const s of all) {
      const cps = Array.from(s.s);
      if (n + cps.length > max) {
        out.push({ kind: s.kind, s: cps.slice(0, Math.max(0, max - n - 1)).join('') + '\u2026' });
        break;
      }
      out.push(s);
      n += cps.length;
    }
    return out;
  });
</script>

{#each segs as s, k (k)}{#if s.kind === 'text'}{s.s}{:else}<span class="ws">{s.s}</span>{#if breaks && s.kind === 'nl'}<br />{/if}{/if}{/each}

<style>
  .ws {
    color: var(--ws);
  }
</style>
