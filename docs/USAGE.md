# Usage

## Global options

| Option | Environment | Default | Meaning |
|---|---|---|---|
| `--home DIR` | `LANTERNFISH_HOME` | `~/.local/share/lanternfish` (Docker: `/data`) | Where traces and caches are kept |
| `-v`, `--verbose` | | off | Show the llama.cpp log |

## `lanternfish probe MODEL [--all] [--threads N]`

Runs one forward pass of 5 tokens and records every node of the compute
graph. It prints:

1. The model summary from the GGUF metadata: architecture, layers, hidden
   size, experts, whether the output head is tied to the embeddings, and
   whether there is a final norm.
2. For each role (residual, attn, ffn, experts, expert_w, shared_gate), the
   node names that were found and on how many layers.
3. With `--all`, every node base name.

It writes the full node list (name, operation, type, shape) to
`$LANTERNFISH_HOME/probe.tsv`.

Use it before the first trace of a new architecture. `residual` must be
found on every layer. If it is not, see
[ARCHITECTURE.md](ARCHITECTURE.md#adding-an-architecture).

## `lanternfish run MODEL --name NAME --prompt TEXT [options]`

| Option | Default | Meaning |
|---|---|---|
| `--name` | required | Trace name: letters, digits, `_`, `.`, `-`; at most 80 characters |
| `--prompt` | required | The user message, or the exact text with `--raw` |
| `-n` | 64 | Tokens to generate after the prompt (stops early at end of turn) |
| `--raw` | off | Do not apply the chat template |
| `--think` | off | For thinking models, keep thinking on (default closes an empty think block) |
| `--threads` | half the cores (`LANTERNFISH_THREADS`) | CPU threads |
| `--labels` | 600 | Number of experts to label, most routing weight first |

Output: one JSON line with the token count, the lens agreement, the number
of expert labels, the time and the trace path. Progress goes to stderr.

A run with an existing name replaces that trace.

## `lanternfish list`

Prints name, model, token count and time for each trace, newest first.

## `lanternfish serve [options]`

| Option | Environment | Default | Meaning |
|---|---|---|---|
| `--host` | | `0.0.0.0` | Address to listen on |
| `--port` | | 8791 | Port |
| `--models-dir D1:D2` | `LANTERNFISH_MODELS` | none (Docker: `/models`) | Directories searched for GGUF files that the UI may trace |
| `--threads` | `LANTERNFISH_THREADS` | half the cores | CPU threads per trace |
| `--labels` | | 600 | Expert labels per trace |

The model list includes `.gguf` files larger than 50 MB, at most 6
directory levels deep. It leaves out `mmproj` and vocabulary-only files,
and all shards after the first. The UI can start traces only of files in
this list.

## Web UI

| Control | Action |
|---|---|
| Trace selector | Opens a trace. The URL keeps the trace, token, view and compare state, so you can share a link. |
| View tabs | Expert route, Logit lens, Write energy, Expert usage. The expert views appear only for mixture-of-experts models. |
| Token strip | Click a token to select it. Prompt tokens are underlined in blue, generated tokens in orange. |
| ← → buttons, arrow keys, Home, End | Step through the tokens |
| Play decode | Animates the generated tokens |
| Compare with | Selects a second trace of the same model. Logit lens and Expert usage then show side by side, or as a difference. |
| New trace | Choose a model, a name and a prompt. The panel shows the run's progress and opens the trace when it is done. |
| Side panel | For the selected token: the next-token candidates, and for each layer the lens top 3, the rank of the next token, and the shared-expert gate |

### Reading the views

- **Expert route.** One row per layer. The x-axis is the expert number.
  Green dots are the experts for the selected token, and the dot area is
  the router weight. Grey dots are the previous token. The blue bar at
  the left is the shared-expert gate. The text at the right is the lens
  top token after that layer. It is green when it equals the token that
  the model generated. A diamond marks full-attention layers in models
  that mix attention types.
- **Logit lens.** Columns are tokens, rows are layers. A bright cell means
  that the generated next token already has a high rank after that layer.
  The row where a column becomes bright is the depth where the model makes
  that prediction.
- **Write energy.** The RMS of what the attention block and the
  feed-forward block add to the residual stream, divided by the RMS of the
  residual stream.
- **Expert usage.** For each layer, how many tokens use each expert, with
  prompt tokens and generated tokens in different colours.

Hover over an expert to see its label: the tokens that its strongest
output direction pushes up and pushes down. A low "rank-1 share" means that
the expert writes many directions, so the label tells you little.

## HTTP API

| Method and path | Request | Response |
|---|---|---|
| `GET /api/health` | | `{status, version, traces, runs_active, model_dirs}` |
| `GET /api/traces` | | `[{name, model{name, arch, n_layers, n_experts}, tokens, created, lens_last_layer_agreement}]`, newest first |
| `GET /api/traces/{name}` | | the trace ([FORMAT.md](FORMAT.md)) |
| `GET /api/models` | | `[{path, name, size_bytes}]` |
| `POST /api/runs` | `{model, name, prompt, n?, raw?, think?}` | `{id}`; 400 with `{error}` if the name, prompt or model is not valid |
| `GET /api/runs` | | `[{id, name, model, state, phase, progress, log_tail, error?}]`; `state` is `queued`, `running`, `done` or `failed` |

Runs are kept in memory and are lost when the server restarts. Traces are
files and stay.

Example:

```sh
curl -s -X POST localhost:8791/api/runs -H 'Content-Type: application/json' \
  -d '{"model":"/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf","name":"api-test","prompt":"Hello","n":16}'
curl -s localhost:8791/api/runs
```
