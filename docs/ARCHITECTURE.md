# Architecture

## Components

```
src/
  main.rs      command line (probe, run, list, serve)
  ffi/         bindgen bindings to llama.h, ggml.h, ggml-backend.h, gguf.h
  gguf.rs      GGUF metadata and tensor rows, dequantized by ggml's to_float
  model.rs     model summary from metadata; ROLES table; output head, final norm, expert weights
  capture.rs   llama.cpp run with an eval callback that keeps the role nodes
  viz.rs       logit lens, write energy, expert labels -> viz.json
  store.rs     trace files, model directory scan
  jobs.rs      trace pipeline and the one-at-a-time run queue
  server.rs    HTTP API (axum) and the embedded UI (rust-embed)
web/           Svelte 5 UI, built to web/dist and embedded in the binary
```

## How a trace is made

1. **Load.** `capture.rs` loads the GGUF with `llama_model_load_from_file`
   on the CPU (no GPU layers). It applies the model's chat template with
   `llama_chat_apply_template`, unless the run is raw.
2. **Callback.** The context is made with `cb_eval` set. llama.cpp calls it
   for every node of the compute graph: first to ask whether the node is
   wanted, then with the computed values. The callback keeps only nodes
   named `<base>-<layer>` where `<base>` is in `ROLES` (`model.rs`).
3. **Reduce at once.** Values are reduced inside the callback, so memory
   does not grow with the graph:
   - residual: the sum of the residual streams (1 for most models, 4 for
     hyper-connection models), and the RMS of each stream,
   - attention and feed-forward outputs: RMS only,
   - expert ids, router weights, shared-expert gate: kept as they are.
4. **Row mapping.** Each `llama_decode` call is one step with a token range.
   Prefill runs in batches of 512 tokens, then decoding runs one token per
   step. A node with as many rows as the step has tokens maps row by row.
   llama.cpp keeps only the rows that produce logits in the last layer, so
   a node with one row maps to the step's last token. The number of
   residual streams comes from layer 0, which is never pruned.
5. **Strided views.** Some nodes are views into a larger tensor. For
   example, `ffn_moe_topk` is the first k columns of an argsort. They are
   read row by row through their strides. (A contiguous-only reader loses
   the routing of every prefill token.)
6. **Logits.** After each step, the top 10 next-token candidates are saved.
   Decoding is greedy.
7. **Lens.** `viz.rs` loads the output head (`output.weight`, or
   `token_embd.weight` when tied) and the final norm (`output_norm.weight`)
   from the GGUF, dequantized to f32. For each layer and token it computes
   `head · (rmsnorm(residual) ⊙ final_norm)`, with a parallel blocked
   SGEMM. It stores the top 5 with probabilities, and the rank of the token
   that the model emitted next.
8. **Expert labels.** For the experts with the most routing weight in the
   trace, it finds the dominant output direction of the expert's down
   projection (6 power iterations on W·Wᵀ). It projects that direction
   through the output head, and keeps the 6 most pushed-up and pushed-down
   tokens. Labels are cached per model in
   `$LANTERNFISH_HOME/expert-labels-<model>.json`.
9. **Save.** `viz.json`, `meta.json` and `prompt.txt` go to
   `$LANTERNFISH_HOME/traces/<name>/`.

## Roles

| Role | Node names, first found wins | Kept as |
|---|---|---|
| residual | `l_out`, `l_last` | summed vector, per-stream RMS |
| attn | `linear_attn_out`, `attn_out`, `attn_output`, `kqv_out` | RMS; the name also gives the layer kind |
| ffn | `ffn_out`, `ffn_moe_out` | RMS |
| experts | `ffn_moe_topk` | expert ids |
| expert_w | `ffn_moe_weights_norm`, `ffn_moe_weights` | weights (layout `{k,T}` or `{1,k,T}`) |
| shared_gate | `shared_expert_gate_sigmoid` | gate value |

A layer is `linear` when its attention node is `linear_attn_out`, and
`attention` for the other names. Nodes of the residual, attn and ffn roles
must have the hidden size as their first dimension. Other shapes are
skipped.

## Adding an architecture

1. Check that llama.cpp loads the model: `lanternfish probe MODEL`. If it
   fails with "unknown model architecture", use a newer llama.cpp.
2. Read the role table in the probe output. If `residual` is "not found",
   run `lanternfish probe MODEL --all` and look in
   `$LANTERNFISH_HOME/probe.tsv` for the node that holds the layer output.
   It usually has the hidden size as `ne0`, one row per token, and an `ADD`
   operation.
3. Add the base name to the right role in `ROLES` (`src/model.rs`). Add a
   linear-attention node name to `LINEAR_MARKERS`.
4. Rebuild and run a short trace. The lens agreement should be above 0.8
   for a dense model with a standard output path.

## Choices

- **CPU only.** The eval callback must copy each node to the host. On the
  CPU this is a memory copy. On a GPU it would make one transfer per node,
  and the graph would be split at each one.
- **ggml for GGUF reading.** `gguf.rs` uses ggml's own `gguf_*` functions
  and dequantizes with ggml's `to_float`. Every quantization type that the
  linked ggml supports works, with no copy of its formats.
- **One run at a time.** A trace uses all CPU threads and maps the whole
  model. Parallel runs would only compete.
- **Reduced capture.** Keeping every node would need tens of GB for a large
  model. The reduced capture needs about tokens × layers × hidden × 4 bytes.
