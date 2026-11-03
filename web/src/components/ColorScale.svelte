<script lang="ts">
  let { gradient, left, mid, right, label }: { gradient: string; left: string; mid?: string; right: string; label: string } =
    $props();
</script>

<span class="scale" role="img" aria-label="{label}: {left} to {right}">
  <span class="end">{left}</span>
  <span class="bar" style:background={gradient}>
    {#if mid}<span class="mid">{mid}</span>{/if}
  </span>
  <span class="end">{right}</span>
</span>

<style>
  .scale {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--dim);
    vertical-align: middle;
  }
  .bar {
    position: relative;
    display: inline-block;
    width: 140px;
    height: 9px;
    border-radius: 2px;
    border: 1px solid var(--line);
  }
  .mid {
    position: absolute;
    left: 50%;
    top: 10px;
    transform: translateX(-50%);
    font-size: 10px;
    line-height: 1;
  }
  .end {
    font-variant-numeric: tabular-nums;
  }
</style>
