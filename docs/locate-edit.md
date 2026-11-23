# Locate-then-edit

Use `lanternfish locate-edit --model MODEL.gguf --request request.json
--cache PRIVATE_DIR --out factors.json --threads 8 --backend cpu`. The command
does not write to `MODEL.gguf`. It returns low-rank factors for
`blk.N.ffn_down.weight`.

The default backend is `cpu`. Use `--backend cuda` or `--backend rocm`
with the matching runtime build to select the first visible device in that
backend. The same option applies to `grad`. The command fails if the device is not
available or a forward or backward graph operation cannot run on it.
The JSON output records the backend and ggml device name in `provenance`.
The factor solve stays on the CPU, and data transfers between the host
and the device are allowed.

### ROCm runtime

Use the patched llama.cpp base and the headless build in [INSTALL.md](../INSTALL.md).
Use a separate build directory for HIP. For a staged ROCm SDK, source its
`env.sh`, then add these options to the llama.cpp configuration:

```sh
-DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_HIP_RCCL=OFF -DGGML_HIP_NO_VMM=ON \
-DGGML_CUDA_FA=OFF -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
-DCMAKE_HIP_COMPILER=/usr/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" \
-DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH" '-DCMAKE_HIP_ARCHITECTURES=gfx1151;gfx1036'
```

The library must include the rocBLAS Tensile data and code objects for the
active device (`lib/rocblas/library`). Patch `0008` adapts the BF16 WMMA
operand types for RDNA3. The edit path uses F32. No new WMMA kernel is needed.
Use `HIP_VISIBLE_DEVICES` to select a device. The ROCm process sets
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (also read by HIP ggml) and
`AMD_OPT_FLUSH=0` before it initializes the runtime.

ROCm 10.1 on the local gfx1036 can read a null device pointer in an
asynchronous pointer-batched GEMM. The standalone case has no ggml code:
one kernel writes the pointer table, then hipBLAS reads it on the same
stream. With the default device-scope fence, the Tensile kernel faults at
address zero. With system-scope fences, all 160 outputs equal 20. This
does not show that gfx1151 has the fault. The edit runtime uses system-scope
fences on both targets. It does not turn on blocking launches.
See the [ROCclr flag definition](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/rocclr/utils/flags.hpp).

To run the standalone check, use a private output directory and an admitted
GPU turn. The `AMD_OPT_FLUSH=1` case can fault; do not run it on a production GPU.

```sh
clang++ -O3 --rocm-path="$ROCM_PATH" --offload-arch=gfx1151 --offload-arch=gfx1036 \
  tools/rocm_batched_gemm.hip -L"$ROCM_PATH/lib" -lhipblas -lamdhip64 -o "$TMPDIR/batch"
AMD_OPT_FLUSH=0 "$TMPDIR/batch"
```

Run the placement smoke from the Lanternfish source tree. `LANTERNFISH_BIN`
must refer to the verified binary. Set `W14B_PLACEMENT_EVIDENCE` to a new
directory. Run the full acceptance with the FP64 witness for the selected
GPU and runtime, not a witness from another GPU:

```sh
W14B_PLACEMENT_BACKEND=rocm python3 -B -m unittest discover -s tools -p test_w14b_placement.py
python3 -B tools/w14b_acceptance.py --backend rocm --threads 6 --binary "$LANTERNFISH_BIN" \
  --evidence "$EVIDENCE" --oracle "$ORACLE" --oracle-sha256 "$ORACLE_SHA256"
```

### Edit request

The request uses this format:

```json
{"method":"rome","layers":[12],"steps":16,"facts":[
  {"subject":"Alice","prompt":"{} lives in","target":"Paris"}
]}
```

Use zero-based layer indices in increasing order. ROME accepts one fact and
layer. MEMIT accepts up to eight facts and two or three layers. AlphaEdit
accepts up to eight facts and one to three layers. Layer choice is an input,
not an automatic localization search. Each prompt must include one `{}` slot.
The lookup position is the last subject token (or the token that overlaps
the end of the subject). Token limits are 128 input and 12 target tokens.

The key is the vector **into** the down projection, after SwiGLU. The
second moment is `C = XX^T` where `X` contains keys from every token in the
fixed corpus, divided by the square root of the token count. The corpus
`data/edit-moment.txt` is original project text under Apache-2.0; it does not
contain evaluation data. SHA-256:
`21ffd89a9f728b1b760433afd385852589cf5012aac10336ac33b2ad23609877`.
The private cache stores the sampled keys by backend, model SHA-256, corpus
SHA-256 and layer.

Patch `0003` extends the W0 F32 backward graph to optimize a displacement at
the subject position. It uses full teacher-forced target cross entropy, a
weak KL term on the clean subject distribution, and norm-limited Adam. It
uses two fixed contexts and a small corpus, not the published generated
context templates or full Wikipedia statistics. Average the two context
keys for each fact before the closed-form solve. Clean MLP output plus the
displacement gives `v*` for ROME. Clean last edit layer residual plus the
displacement gives `z*` for MEMIT and AlphaEdit.

- **ROME:** Solve for `a = (C + ridge I)^-1 mean(k)`, then normalize `a` by
  its dot product with the canonical key. The update is `(v* - Wk) a^T`.
- **MEMIT:** At each layer, solve
  `R K^T (32 C + KK^T + ridge I)^-1`. Recompute activations after each
  prior layer update. Divide the remaining `z* - z` error by the number
  of remaining layers.
- **AlphaEdit:** At each layer, project the edit keys into the null space
  of the preserved sample keys. Solve
  `R (PK)^T ((PK)(PK)^T + ridge I)^-1` in that space. This is the first-batch
  solution; it does not keep the reference code's preceding-batch matrix.
  Null-space protection is exact for the sampled key span before output
  rounding. It does not guarantee unchanged logits on other prompts.

The implementation uses Woodbury with a small F64 Gram solve. The ridge
uses the mean diagonal of `C` with a positive floor (ROME uses 1% of that
scale). For a BF16 GGUF, the CLI expands weights to F32 in a temporary file
and loads a private writable copy. The CLI removes the temporary file
after load. It rounds each intermediate multi-layer update once to BF16
before the next layer's forward pass. F32 source GGUF uses an F32 private
copy. Quantized edit inputs are not supported.

References: [Meng et al., ROME (2022)](https://arxiv.org/abs/2202.05262),
[Meng et al., MEMIT (2023)](https://arxiv.org/abs/2210.07229), and
[Fang et al., AlphaEdit (2024)](https://arxiv.org/abs/2410.02355).
