# Docker

## Image

The `Dockerfile` has four stages:

| Stage | Base | Does |
|---|---|---|
| `llama` | `debian:trixie-slim` | Clones llama.cpp at `LLAMA_CPP_REF` and builds the shared libraries for the CPU |
| `web` | `node:22-bookworm-slim` | `npm ci` and `npm run build` for the UI |
| `rust` | `rust:1-trixie` | Builds `lanternfish` against stage 1 and embeds the UI from stage 2 |
| final | `debian:trixie-slim` | The binary, the llama.cpp libraries, `libgomp1` and `curl` (for the health check) |

The final image runs as the user `lanternfish` (uid 10001).

## Build arguments

| Argument | Default | Meaning |
|---|---|---|
| `LLAMA_CPP_REPO` | `https://github.com/ggml-org/llama.cpp` | llama.cpp source |
| `LLAMA_CPP_REF` | `d81235049384534c167caea52b85a694f6103d14` | Commit to build. This is upstream build 11429 (2026-10-05), which supports `qwen4exp`. Use a newer commit for newer architectures. |

llama.cpp is built with `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON` and
`GGML_CPU_ALL_VARIANTS=ON`. The image therefore runs on any x86-64 CPU. At
start, ggml loads the best CPU kernels (AVX2, AVX-512 and others) for the
host.

## Volumes and environment

| Path or variable | Default | Meaning |
|---|---|---|
| `/models` | | Mount your GGUF directory here, read-only. `LANTERNFISH_MODELS=/models`. |
| `/data` | volume | Traces and expert-label caches. `LANTERNFISH_HOME=/data`. |
| `LANTERNFISH_THREADS` | half the cores | CPU threads per trace |

The models must be readable by uid 10001. If they are not (for example a
model directory with mode 700), run as your own user and use a directory
for `/data`:

```sh
mkdir -p ~/lanternfish-data          # create it first; see below
docker run -d --name lanternfish --user "$(id -u):$(id -g)" -p 8791:8791 \
  -v /path/to/models:/models:ro -v ~/lanternfish-data:/data lanternfish
```

Create a bind-mounted directory before `docker run`. If it does not
exist, Docker creates it as root, and lanternfish fails at start with
"Permission denied (os error 13)". A named volume (`-v
lanternfish-data:/data`) does not have this problem.

## Run

```sh
docker run -d --name lanternfish --restart unless-stopped \
  -p 8791:8791 -v /srv/models:/models:ro -v lanternfish-data:/data lanternfish
```

One trace from the command line, with no server:

```sh
docker run --rm -v /srv/models:/models:ro -v lanternfish-data:/data lanternfish \
  run /models/model.gguf --name cli-test -n 16 --prompt "Hello"
```

Compose: see `compose.yaml`. Set `LANTERNFISH_MODELS_DIR` to the model
directory.

## Numbers compared with a native build

The image selects CPU kernels for the host at run time (for example
`zen4`). A native llama.cpp build can use other kernels. Floating-point
results then differ slightly. Measured on the same model and prompt
(Qwen3.5-4B): lens agreement 0.901 in the image and 0.887 natively, with
the same generated tokens. Compare traces made by the same build.

## Health

`HEALTHCHECK` calls `GET /api/health` every 30 s. `docker ps` shows
`healthy` when the server answers.

## Memory

The container memory-maps the model files. Page cache that the kernel
uses for those files counts toward a container memory limit (`--memory`).
For large models, do not set a limit, or set it above the model size.

## GPU

The image is CPU only. lanternfish uses the CPU because the capture
callback copies each graph node to the host (see
[ARCHITECTURE.md](ARCHITECTURE.md#choices)). A GPU build would not make
traces faster.
