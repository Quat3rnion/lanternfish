# traverse web UI

This directory contains the browser UI for traverse. The UI shows how a model
routes each token through its layers and experts during prefill and decode.

The stack is Vite, Svelte 5 (runes) and TypeScript. The UI uses no UI
framework and no chart library. The heavy views use the canvas 2D API.

## Develop

```sh
npm install
npm run dev        # http://localhost:5179, mock API from fixtures/
npm run check      # svelte-check, must report 0 errors and 0 warnings
npm run build      # writes dist/
```

To use a running traverse server instead of the mock API, set
`TRAVERSE_API`. Vite then sends all `/api` requests to that server:

```sh
TRAVERSE_API=http://127.0.0.1:8791 npm run dev
```

Do not use port 3000 or port 4095.

## Build output

`npm run build` writes `dist/index.html` and `dist/assets/*`. All asset
paths are relative (`base: './'`), and the app requests the API with relative
URLs (`api/...`). The Rust server embeds `dist/` and serves it at `/`.

## Mock API

`mock/api.ts` is a Vite plugin. It runs only in `npm run dev` and is not part
of the build. It implements the API contract:

| Request | Mock behavior |
| --- | --- |
| `GET /api/traces` | Lists every `fixtures/*.json`, plus `<moe>-variant`, newest first. |
| `GET /api/traces/{name}` | Returns the fixture. `<moe>-variant` is a copy with changed routes and ranks, so that compare mode has a second trace of the same model. |
| `GET /api/models` | One model for each fixture model. |
| `POST /api/runs` | Validates the body and starts a fake run. A prompt that contains the word `fail` makes the run fail at 50%. |
| `GET /api/runs` | Fake progress: about 1 s queued, then about 9 s running. When a run is done, a copy of the matching fixture becomes a new trace. |

Fixtures:

- `fixtures/signal-fib.json`: Signal 3.8 Flash Next, 48 layers, 512 experts,
  top-10, hybrid attention (linear and full).
- `fixtures/llama32-3b-fib.json`: Llama 3.2 3B, 28 layers, dense.

## URL state

The query string holds the view state, so a link opens the same view:
`?t=<trace>&tok=<token index>&view=route|lens|write|usage&cmp=<trace>&cmode=side|diff`.

## Layout

```
index.html
vite.config.ts            Vite config, mock API or proxy
mock/api.ts               dev-only mock of /api
fixtures/                 sample traces for the mock
src/main.ts               mounts App
src/app.css               theme tokens and base control styles
src/App.svelte            header, trace picker, tabs, compare controls,
                          URL state, play, run polling
src/components/
  TokenStrip.svelte       token list (listbox), prefill and decode styling
  ViewPane.svelte         legend and canvas(es) for the active view and compare mode
  Canvas.svelte           devicePixelRatio-correct canvas, resize, hover, click
  Tooltip.svelte          shared tooltip for all canvases
  DetailPanel.svelte      selected token: next-token candidates, per-layer table
  NewTraceDialog.svelte   start a run, run list with progress and log tail
  Piece.svelte            token text with visible whitespace (· ↵ →)
  ColorScale.svelte       color scale legend
src/lib/
  types.ts                API types
  api.ts                  fetch wrapper for /api
  analysis.ts             per-trace derived data (typed arrays), computed once
  colors.ts               palette, sequential and diverging color scales
  format.ts               text and number helpers
  url.ts                  read and write the query string
  tooltip.svelte.ts       shared tooltip state
  views/painter.ts        Painter interface between Canvas and the views
  views/route.ts          Expert route view
  views/grid.ts           Logit lens, Write energy, lens difference heatmaps
  views/usage.ts          Expert usage view and usage difference
  views/tips.ts           tooltip text for experts and cells
```

## Views

- **Expert route** (MoE only): one row for each layer, x is the expert id.
  Green dots are the experts of the selected token. Grey dots are the experts
  of the previous token. Dot area shows the router weight.
- **Logit lens**: tokens x layers. The color shows the rank of the emitted
  next token in the logit lens after the layer.
- **Write energy**: tokens x layers. The color shows the RMS that the layer
  writes into the residual, divided by the residual RMS. The top of the scale
  is the 99th percentile.
- **Expert usage** (MoE only): the number of tokens that each expert received,
  split into prefill and decode.

The UI hides the MoE views when `n_experts` is 0. It marks full-attention
layers with ◆ only when the model has more than one layer kind.

## Compare mode

Select a second trace in "Compare with". The list shows only traces with the
same `arch`, `n_layers` and `n_experts`. Compare mode applies to the Logit lens
and Expert usage views:

- **Side by side**: the two views next to each other.
- **Difference**: A minus B. For the logit lens, tokens are aligned by
  position; red ticks mark positions where the two traces have different
  tokens. For expert usage, the UI compares the share of routed tokens, so
  traces with different token counts can be compared.

## Keyboard

- Left and Right arrow keys: previous and next token. Home and End: first and
  last token.
- Tab moves through the controls. In the view tabs, the arrow keys change the view.
- The token strip is a listbox and has a visible focus ring.
