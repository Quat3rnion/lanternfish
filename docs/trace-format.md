# CPU trace and gradient exports, schema version 1

`trace` runs a raw prompt through llama.cpp on the CPU. It does not apply a
chat template or generate tokens. `grad` computes a next-token gradient on the
CPU. These commands do not change the GGUF or use the Lanternfish trace store.
Build with the [pinned llama.cpp patch series](../patches/README.md).

## Capture a trace

Save this example as `spec.json`:

```json
{
  "layers": [0, 1],
  "roles": ["l_out", "attn_out", "ffn_inp", "ffn_out", "kq_soft_max"],
  "positions": [1, 2],
  "top_k": 5,
  "byte_cap": 1048576
}
```

```sh
lanternfish trace --model model.gguf --prompt "The capital of France is" \
  --spec spec.json --out trace-1
```

- `layers` and `positions` are zero-based. Positions refer to the tokenized
  prompt, including any special tokens such as BOS. Duplicate selections
  count once. A request must select at least one position.
- `roles` lists exact graph node names without the `-<layer>` suffix.
  `l_out` is the residual after the block. `attn_out` is the projected
  attention write. `ffn_inp` is the FFN input residual before FFN
  normalization. `ffn_out` is the FFN write before the residual add.
  `kq_soft_max` is the attention probability tensor when exposed by the graph.
- `top_k` selects the number of logits to keep at each position. It also
  enables a logit lens on dense Llama and Qwen2 for each selected layer and position with `l_out`,
  even if `roles` does not include `l_out`. Set `top_k` to `0` to skip logits.
  Set `layers` and `roles` to `[]` for model logits only. `top_k` cannot
  exceed the vocabulary size.
- `byte_cap` is the maximum for the manifest and all binary payloads together.
  The command fails instead of discarding requested data to fit the cap. The
  cap is not a limit on model memory, ggml work buffers, or projection memory.

For Qwen2MoE and Qwen3MoE, select `ffn_moe_logits` for raw router scores
(`shape: [n_experts]`). Select `ffn_moe_topk` and `ffn_moe_route_weights`
for the selected expert IDs and final mixing weights in the same slot order
(`shape: [experts_per_token]`). IDs are exact integers stored as `f32_le`.
These roles are read-only. `top_k` still controls model output logits, not
the number of experts. MoE traces do not have a logit lens.

## Intervene during a trace

Save a JSON array as `edit.json`, for example:

```json
[
  {"layer": 0, "role": "attn_out", "positions": [1], "op": "scale", "factor": 0.5},
  {"layer": 0, "role": "ffn_out", "positions": [2], "op": "zero"}
]
```

```sh
lanternfish trace --model model.gguf --prompt "The capital of France is" \
  --spec spec.json --intervention edit.json --out trace-edit
```

Use `add` or `replace` with a `vector` array of finite numbers, one per hidden
dimension. `scale` multiplies the row by a finite `factor`. `zero` sets all
values in the row to zero. `kq_soft_max` is read-only. Rules can target rows
outside the capture spec. Two rules cannot edit the same layer, role, and
position. The command fails if it cannot apply any requested edit.

Lanternfish changes the F32 activation after the node runs and before its
consumers run. Capture records the value **after** the edit. An edit can affect
later nodes and tokens through normal data flow. It does not change an
attention calculation or KV write that has already run.

For a Qwen2MoE or Qwen3MoE expert ablation, use
`{"layer":0,"role":"ffn_out","positions":[1],"op":"zero_expert","expert":3}`.
Lanternfish sets the selected expert's weighted contribution to zero before
the expert sum. It does not select a replacement or normalize the other
weights again. It makes no change at positions that did not select the
specified expert. The manifest records `expert` for each requested position.

## Trace file layout

The output directory must not exist before the command runs. Read an export
only if `manifest.json` exists. The command writes it after all payloads.

`manifest.json` is UTF-8 JSON. The numeric `schema_version` is `1` and `kind`
is `"trace"`. Other fields:

| Field | Contents |
|---|---|
| `model` | GGUF metadata: `arch`, `name`, `n_layers`, `hidden`, `n_experts`, `top_k` (experts per token), `tied_head`, `final_norm`, and input `path`. The path is not a content ID. |
| `spec` | Capture spec with selections sorted and deduplicated. |
| `tokens` | Full prompt token sequence as `{position, token_id}` records. |
| `tensors` | Payload descriptors, one per captured `(layer, role, position)`, with those three fields and `node`, the full graph node name. |
| `missing` | Requested `(layer, role, position)` records that the graph did not expose. |
 | `interventions` | `(layer, role, position)` records for applied edits. Expert ablations also give `expert`. Keep the input intervention file to reproduce the edit. |
| `model_logits` | `{position, top}` records from llama.cpp. |
| `logit_lens` | `{layer, position, top}` records from projecting `l_out` through the model's RMS norm and output head, including Qwen2's optional output bias. If a model ties its head, the projection uses its token embeddings. Missing `l_out` nodes have no lens record. |

Each `top` is a list of `{token_id, logit}` records. Values are raw logits,
not probabilities. The list is sorted by decreasing logit, then by increasing
token ID for ties.

Each payload descriptor has `file`, `shape`, `dtype`, `bytes`, and `sha256`.
`file` is a relative file name such as `tensor-000000.f32`; `dtype` is
`"f32_le"`. The file contains only little-endian IEEE 754 binary32 samples,
in row-major order (last dimension varies fastest). `bytes` equals four
times the product of `shape`. `sha256` is the lowercase hex SHA-256 of the
exact file bytes.

An activation row has shape `[hidden]`. An attention row for one query
position has shape `[heads, key_slots]`. The second axis is the native KV
cache slot axis, including any padding and masked slots. There is one
sequence, no cache eviction, and slots contain prompt tokens in insertion
order for dense Llama and Qwen2. Save the shape; do not assume the axis
length equals the number of prompt tokens.

## Compute a gradient

```sh
lanternfish grad --model model-f32.gguf --prompt "The capital of France is" \
  --layer 0 --target-token 123 --out grad-1 --backend cpu
```

Replace `123` with the model vocabulary ID of the target token. The loss
is the negative log probability of that token at the last prompt position.
`--layer` selects `blk.<layer>.ffn_down.weight` in a dense Llama or Qwen2
model. The backward graph uses the original layer stack, with direct K/V
instead of the inference KV cache. It propagates through subsequent attention
and FFN blocks. It does not run an optimizer or update the model.
The default backend is `cpu`. Use `--backend cuda` to select the first visible
CUDA device. The command fails if no CUDA device is available or if a
forward or backward graph operation cannot run on that device. Data transfers
between the host and the device are allowed.

`grad-1/manifest.json` has `schema_version: 1` and `kind: "gradient"`. It
records `model`, `tokens` (an array of token IDs in order), `position`,
`target_token`, `objective` (`"next_token_cross_entropy"`), `loss`, `layer`,
`weight` (the GGUF tensor name), `tensor` (the payload descriptor), and
`provenance` (`backend` and the ggml `device` name).
`gradient.f32` has shape `[hidden, ffn_width]`, with the input dimension
varying fastest, like the corresponding GGUF weight.

## Limits

- Trace uses one CPU sequence, F32 K/V and no flash attention. It accepts up
  to 4096 prompt tokens, or the model's training context if smaller. It
  captures only selected rows. Other architectures can expose different
  graph nodes or layouts; unsupported layouts fail, and absent nodes are
  listed in `missing`.
- Write-back and logit lens support Llama and Qwen2. Trace and write-back
  work with BF16 and quantized GGUF weights when those graph nodes use F32
  activations. The lens reads the dequantized head, so its values can differ
  from native logits, especially with a quantized head.
- Gradient requires all F32 model weights. It supports a dense Llama or
  Qwen2 model with ordinary causal attention, and one prompt of at most
  256 tokens (or the model's training context if smaller) in one ubatch.
  It exports one FFN down-weight gradient per call. It does not support
  BF16 or quantized backward, activation gradients, or MoE. A full
  backward pass can need substantially more memory than inference.
