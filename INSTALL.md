# INSTALL (for agents)

Do the steps in order. Each step has a check. Stop if a check fails.

## 1. Requirements

- Linux, Rust 1.85 or later (edition 2024), Node.js 20 or later with npm.
- libclang (for bindgen): `ls /usr/lib/libclang.so*`.
- llama.cpp installed as shared libraries with headers:
  `ls /usr/include/llama.h /usr/include/gguf.h /usr/lib/libllama.so`.

If llama.cpp is not installed, build it from source:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
```

Check: the llama.cpp version supports the model.
`llama-cli -m MODEL.gguf -p hi -n 1` loads without "unknown model
architecture".

## 2. Build the UI

```sh
npm --prefix web ci
npm --prefix web run build
```

Check: `web/dist/index.html` exists. The binary embeds `web/dist`, so build
the UI before the binary.

## 3. Build the binary

System llama.cpp:

```sh
cargo build --release
```

Source build of llama.cpp:

```sh
LLAMA_INCLUDE_DIR=~/llama.cpp/include:~/llama.cpp/ggml/include \
LLAMA_LIB_DIR=~/llama.cpp/build/bin cargo build --release
```

Check: `target/release/traverse --version` prints the version.
`cargo test --release` passes.

On hosts where `/tmp` is not allowed, set `TMPDIR` and `CARGO_TARGET_DIR`
to an allowed directory first.

## 4. Probe the model

```sh
target/release/traverse probe MODEL.gguf
```

Check: `residual` is found on every layer, and `attn` and `ffn` are found.
For a MoE model, `experts` and `expert_w` are found too.

## 5. Trace and view

```sh
export TRAVERSE_HOME=/path/with/space/traverse
target/release/traverse run MODEL.gguf --name test -n 32 --prompt "Write a haiku about rivers."
target/release/traverse serve --port 8791 --models-dir /path/to/models
```

Check: the `run` output has `lens_agreement` above 0.8 for a standard
dense model. `curl -s localhost:8791/api/traces` lists `test`.

Keep it running as a user service:

```sh
systemd-run --user --unit=traverse -E TRAVERSE_HOME=$TRAVERSE_HOME \
  $PWD/target/release/traverse serve --port 8791 --models-dir /path/to/models
```

Do not use ports 3000 or 4095 on the operator's hosts.

## Time and memory

- The GGUF is memory-mapped. For good speed, free RAM should be about the
  file size.
- Decode time per token is about one full read of the active weights.
- The LM head is held as f32.
- Expert labels: about 20 s for 600 experts of a 125B MoE.
