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

Check out the llama.cpp base and apply the
[required patches](patches/README.md) before building. Run these commands from
the Lanternfish source directory:

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
git -C ../llama.cpp checkout --detach 184f6536024668e8908ab8c86033d56708e62b40
sh patches/apply-llama.sh ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build ../llama.cpp/build --target llama -j 4
```

Check: the llama.cpp version supports the model architecture. Older
versions fail with "unknown model architecture" in `lanternfish probe`.

### B3. Build

Set the library paths from the Lanternfish source directory:

```sh
export LLAMA_INCLUDE_DIR="$PWD/../llama.cpp/include:$PWD/../llama.cpp/ggml/include"
export LLAMA_LIB_DIR="$PWD/../llama.cpp/build/bin"
```

For the standalone app, build the UI before the binary:

```sh
npm --prefix web ci
npm --prefix web run build
cargo build --release --locked
```

If `/tmp` is not allowed, set `TMPDIR` and `CARGO_TARGET_DIR` to an allowed
directory first.

Check: `target/release/lanternfish --version` prints the version, and
`cargo test --release --locked` passes. To run the CPU CLI regressions
with Python 3 and generated GGUF fixtures:

```sh
LANTERNFISH_BIN="$PWD/target/release/lanternfish" \
  python3 -B -m unittest discover -s tests -v
```

The binary embeds `web/dist`. If you see
the page "UI not built", build the UI and then rebuild the binary.

### Headless CLI build

Set `LLAMA_INCLUDE_DIR` and `LLAMA_LIB_DIR` as above. Skip the Node.js and
Svelte steps. Use a separate target directory to keep both release binaries:

```sh
cargo build --release --locked --no-default-features --target-dir target/headless
cargo test --release --locked --no-default-features --target-dir target/headless
```

Use `target/headless/release/lanternfish` for Model Brewery. This binary has no
embedded web assets, HTTP listener, or `serve` command. `trace` and `grad` use
only the output directory you give them. Legacy `run` still uses `--home` or
`LANTERNFISH_HOME` and writes `viz.json`. The headless test uses Python 3 to
generate small model fixtures and run the CLI. Set `TMPDIR` to a writable
temporary directory if necessary.

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
