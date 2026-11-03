# traverse

`traverse` shows how a language model processes a prompt, token by token
and layer by layer, in prompt processing (prefill) and in decoding. It works
with any model that llama.cpp can load from a GGUF file. It has no
model-specific code.

For each token and layer it records:

- the experts that the router selects and their weights (mixture-of-experts
  models),
- how much the attention block and the feed-forward block write into the
  residual stream,
- the logit lens: the tokens that the residual stream already predicts after
  the layer, and the rank of the token that the model actually emits.

A web UI shows the traces and starts new ones.

The backend is Rust (FFI to the installed llama.cpp). The UI is Svelte. One
binary serves both.

## Tested models

| Model | Architecture | Type | Lens agreement at last layer | Trace time |
|---|---|---|---|---|
| Llama 3.2 3B Instruct Q4_K_M | `llama` | dense | 1.000 | 6 s (70 tokens) |
| Qwen3.5-4B Q4_K_M | `qwen35` | dense, hybrid linear/full attention | 0.887 | 6.8 s (72 tokens) |
| Signal-3.8-Flash-Next Q8_0 (125B) | `qwen4exp` | 512-expert MoE, hybrid attention, 4 hyper-connection streams | 0.969 | 180 s (56 tokens, 16 threads) |

"Lens agreement" is the share of positions where the lens top-1 token at
the last layer equals the token that the model emitted. A value near 1
shows that the lens reads the residual stream correctly. Models with an
extra output mixer (hyper-connections) or hybrid layers give lower values.
The rank of the next token is still useful for them.

The Rust output was compared field by field with the earlier Python
version on the same models. Token ids, expert ids, router weights, layer
kinds and lens top-1 were identical in every cell. The next-token rank was
identical in 98.5% of cells. The other cells differ by float rounding at
deep ranks.

## Quick start

```sh
# 1. UI
npm --prefix web ci && npm --prefix web run build
# 2. binary (links the system llama.cpp; see INSTALL.md for a source build)
cargo build --release
# 3. use it
target/release/traverse probe MODEL.gguf
target/release/traverse run MODEL.gguf --name first --prompt "Write a Python function that reverses a list."
target/release/traverse serve --port 8791 --models-dir /path/to/models
```

Open `http://HOST:8791/`.

Traces are in `$TRAVERSE_HOME/traces/<name>/` (default
`~/.local/share/traverse`).

## Commands

| Command | Does |
|---|---|
| `traverse probe MODEL [--all]` | One small forward pass. Prints the model summary and the graph nodes found for each role. `--all` lists every node base name. Writes `probe.tsv`. |
| `traverse run MODEL --name N --prompt P [-n 64] [--raw] [--think] [--threads T] [--labels 600]` | Prefill plus `n` greedy decode tokens, then the lens and the expert labels. Applies the model's chat template unless `--raw`. For thinking models it closes an empty think block unless `--think`. |
| `traverse list` | Lists the traces. |
| `traverse serve [--host 0.0.0.0] [--port 8791] [--models-dir D1:D2] [--threads T]` | Serves the UI and the API. The UI can start traces of the GGUF files under `--models-dir` only. Runs go one at a time. |

Global options: `--home DIR` (or `TRAVERSE_HOME`), `-v` to show the
llama.cpp log.

## UI

| View | Shows | Models |
|---|---|---|
| Expert route | the selected token's experts in every layer (dot size = router weight), the previous token in grey, the shared-expert gate, the lens token per layer | MoE |
| Logit lens | tokens x layers: bright when the emitted next token is already near rank 0 | all |
| Write energy | tokens x layers: (attention + feed-forward write RMS) / residual RMS | all |
| Expert usage | per layer, how many tokens use each expert, prefill against decode | MoE |
| Compare | two traces of the same model side by side or as a difference | all |
| New trace | choose a model, write a prompt, follow the run | all |

See [web/README.md](web/README.md) for the UI code.

## API

| Method and path | Returns |
|---|---|
| `GET /api/traces` | `[{name, model{name, arch, n_layers, n_experts}, tokens, created}]`, newest first |
| `GET /api/traces/{name}` | the trace (`viz.json`, see [docs/FORMAT.md](docs/FORMAT.md)) |
| `GET /api/models` | `[{path, name, size_bytes}]` GGUF files under `--models-dir` |
| `POST /api/runs` | body `{model, name, prompt, n, raw, think}`; returns `{id}` |
| `GET /api/runs` | `[{id, name, model, state, phase, progress, log_tail, error?}]` |

The server has no authentication. Bind it to a trusted network only.

## How it works

1. `src/capture.rs` loads the GGUF through `libllama` on the CPU and sets
   the scheduler evaluation callback. The callback reads the graph nodes
   named `<base>-<layer>` for the bases in `ROLES` (`src/model.rs`). It
   keeps only reduced values: the summed residual, write RMS, expert ids,
   weights and gates.
2. llama.cpp prunes the last layer to the rows that produce logits. The
   collector maps those rows to the last prompt token or the decode token.
   The number of residual streams comes from layer 0, which is never pruned.
3. `src/viz.rs` computes the lens: `lm_head(rmsnorm(sum of streams) *
   output_norm)`, with a blocked parallel SGEMM. Expert labels use 6 power
   iterations on the expert's down projection, then the LM head.
4. `src/gguf.rs` reads the GGUF with ggml's own `gguf_*` functions and
   dequantizes with ggml's `to_float`. So every quant type that the
   installed ggml supports also works here.

### Role names

| Role | llama.cpp node names (first found wins) |
|---|---|
| residual stream after the layer | `l_out`, `l_last` (multi-stream; streams are summed) |
| attention write | `linear_attn_out`, `attn_out`, `attn_output`, `kqv_out` |
| feed-forward write | `ffn_out`, `ffn_moe_out` |
| selected experts | `ffn_moe_topk` |
| expert weights | `ffn_moe_weights_norm`, `ffn_moe_weights` |
| shared-expert gate | `shared_expert_gate_sigmoid` |

If `traverse probe` shows "not found" for `residual`, run `probe --all`
and add the node name to `ROLES` in `src/model.rs`. The node must have the
hidden size as its first dimension.

## Limits

- CPU only, greedy decoding.
- The trace shows the GGUF that you give it, in llama.cpp. It does not show
  another runtime or another quantization of the same model.
- The chat template is applied with llama.cpp's built-in formats, not with
  a full Jinja engine. Some templates render differently from the model's
  own Jinja (for example, Llama 3.2 without the date header). Use `--raw`
  to give the exact text.
- Lens probabilities are calibrated only when the output path is "final
  norm, then LM head".
- Memory: the LM head is held as f32 (vocabulary x hidden x 4 bytes, 2.5 GB
  for a 248k x 2560 head). The summed residual is held per token and layer.

## Licence

Apache-2.0.
