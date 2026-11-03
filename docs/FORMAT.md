# Trace format (`viz.json`)

One JSON object per trace, in `$LANTERNFISH_HOME/traces/<name>/viz.json`.
`meta.json` beside it is a short summary for listings. `prompt.txt` holds
the exact text that was tokenized.

| Field | Type | Meaning |
|---|---|---|
| `name` | string | trace name |
| `model` | object | `arch, name, n_layers, hidden, n_experts, top_k, tied_head, final_norm, path` |
| `n_layers`, `n_experts`, `top_k` | int | copies of the model fields (`n_experts` 0 = dense) |
| `tokens` | array | `{i, id, piece, phase}`; `phase` is `prefill` or `decode` |
| `steps` | array | one per `llama_decode` call: `{first, n, phase}` (token index range) |
| `next_candidates` | array | per step: top-10 `{id, piece, p}` after the step's last token |
| `layers` | array | per layer: `{layer, kind, tokens}` |
| `layers[].kind` | string | `attention`, `linear` (recurrent/linear attention) or `unknown` |
| `layers[].tokens` | object | key = token index (string), value = cell (below) |
| `experts` | object | key `"layer.expert"`: `{pos[6], neg[6], rank1}` output-direction label |
| `lens_last_layer_agreement` | float | share of positions where the last-layer lens top-1 = emitted token |
| `lens_note` | string | how the lens is computed |
| `captured` | array | graph node bases that were found |
| `created` | string | UTC ISO time |
| `prompt` | string | the tokenized text |
| `tool` | string | `lanternfish <version>` |

## Cell

| Field | Present | Meaning |
|---|---|---|
| `lens` | always | top-5 `[piece, probability]` of the logit lens after the layer |
| `resid_rms` | always | RMS of the summed residual |
| `write.attn`, `write.ffn` | if found | RMS written by the attention and feed-forward blocks |
| `streams_rms` | multi-stream models | RMS of each residual stream |
| `next_rank` | not for the last token | rank of the emitted next token in the lens (0 = top) |
| `experts`, `weights` | MoE | selected expert ids and router weights (after renormalization if the model does it) |
| `shared_gate` | shared-expert models | shared-expert gate value (0 to 1) |

The last layer has cells only for the last prompt token of each prefill
step and for the decode tokens, because llama.cpp prunes the other rows
there.
