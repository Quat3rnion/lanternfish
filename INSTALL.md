# Install

Do the steps in order. Each step ends with a check. If a check fails, stop
and read the output before you continue. The steps are also written so that
a script or a coding agent can follow them.

Choose one method:

- **A. Docker** (recommended): you do not need a local toolchain.
- **B. From source**: use this when the host already has llama.cpp, or when
  Docker is not available.

Before you start, find these values:

1. The directory that contains your GGUF models. These steps call it
   `MODELS_DIR`.
2. The port to use (default 8791). Make sure that no other program uses it
   (`ss -ltn | grep :8791`).
3. On some hosts, `/tmp` is small or not permitted. If so, select a
   different directory for temporary build files.

lanternfish runs on Linux x86-64. Other platforms are not tested.

## A. Docker

### A1. Check Docker

```sh
docker version --format '{{.Server.Version}}'
```

Check: the command prints a version. If permission is denied, your user is
not in the `docker` group. Add it (`sudo usermod -aG docker $USER`, then log
in again), or ask the administrator of the host.

### A2. Get the source and build the image

```sh
git clone https://github.com/Quat3rnion/lanternfish
cd lanternfish
docker build -t lanternfish .
```

The build compiles llama.cpp (5 to 15 minutes). To build another llama.cpp
version, add `--build-arg LLAMA_CPP_REF=<commit or tag>`.

Check: `docker run --rm lanternfish --version` prints `lanternfish 0.x.y`.

### A3. Start the server

```sh
docker run -d --name lanternfish --restart unless-stopped \
  -p 8791:8791 \
  -v MODELS_DIR:/models:ro \
  -v lanternfish-data:/data \
  lanternfish
```

Check:

```sh
sleep 3
curl -fsS http://127.0.0.1:8791/api/health
curl -fsS http://127.0.0.1:8791/api/models | head -c 300
```

The first command returns `"status":"ok"`. The second lists GGUF files
from `MODELS_DIR`. If the list is empty, the mount is wrong, or the files
are smaller than 50 MB, or they are `mmproj` files.

### A4. Make a test trace

Use the smallest model from `/api/models`:

```sh
docker exec lanternfish lanternfish probe /models/SMALL_MODEL.gguf
docker exec lanternfish lanternfish run /models/SMALL_MODEL.gguf \
  --name install-check -n 16 --prompt "Write a haiku about rivers."
```

Check: `probe` shows `residual` found on every layer. `run` prints JSON with
`"lens_agreement"`. For a standard dense model, the value is above 0.8.

### A5. Open the UI

Open `http://HOST:8791/` in a browser and select the trace `install-check`.

## B. From source

### B1. Requirements

```sh
cargo --version          # 1.88 or later
node --version           # 20 or later
ls /usr/lib/libclang.so* /usr/lib/*/libclang*.so* 2>/dev/null | head -1
```

Check: all three commands print something. Install a missing part with the
system package manager. The package `clang` or `libclang-dev` gives
libclang. Use [rustup](https://rustup.rs) for a current Rust.

### B2. llama.cpp

Check for a system install:

```sh
ls /usr/include/llama.h /usr/include/gguf.h && ldconfig -p | grep -q libllama && echo found
```

If it is not found, build it:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j
```

Check: the llama.cpp version supports the model architecture. Older
versions fail with "unknown model architecture" in `lanternfish probe`.

### B3. Build

```sh
git clone https://github.com/Quat3rnion/lanternfish
cd lanternfish
npm --prefix web ci
npm --prefix web run build
```

System llama.cpp:

```sh
cargo build --release
```

Source-built llama.cpp:

```sh
LLAMA_INCLUDE_DIR=$HOME/llama.cpp/include:$HOME/llama.cpp/ggml/include \
LLAMA_LIB_DIR=$HOME/llama.cpp/build/bin \
cargo build --release
```

If `/tmp` is not allowed, set `TMPDIR` and `CARGO_TARGET_DIR` to an allowed
directory first.

Check: `target/release/lanternfish --version` prints the version, and
`cargo test --release` passes.

Build the UI before the binary. The binary embeds `web/dist`. If you see
the page "UI not built", build the UI and then rebuild the binary.

### B4. Start the server

```sh
export LANTERNFISH_HOME=$HOME/.local/share/lanternfish
target/release/lanternfish serve --port 8791 --models-dir MODELS_DIR
```

To keep it running on a systemd host:

```sh
systemd-run --user --unit=lanternfish -E LANTERNFISH_HOME=$LANTERNFISH_HOME \
  $PWD/target/release/lanternfish serve --port 8791 --models-dir MODELS_DIR
```

`systemd-run` units do not start again after a reboot. For a permanent
service, write a user unit file and run `systemctl --user enable --now`.

Check: same as A3 and A4, with `target/release/lanternfish` in place of
`docker exec lanternfish lanternfish`.

## Resources

| Item | Need |
|---|---|
| RAM | about the GGUF file size (the file is memory-mapped), plus vocabulary x hidden size x 4 bytes for the output head |
| CPU | all cores are used during a trace; `--threads` or `LANTERNFISH_THREADS` limits this |
| Disk | about 1 MB per 50 traced tokens |

Only one trace runs at a time. Later requests wait in a queue.

## Remove

Docker: `docker rm -f lanternfish && docker volume rm lanternfish-data`.
Source: stop the service and delete `LANTERNFISH_HOME`.
