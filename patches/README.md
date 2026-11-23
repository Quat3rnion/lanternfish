# llama.cpp patch series

Base: `184f6536024668e8908ab8c86033d56708e62b40` from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

Apply the patches to a clean checkout of this commit:

```sh
sh patches/apply-llama.sh /path/to/llama.cpp
```

1. `0001-name-dense-trace-nodes.patch`: Name the projected Qwen2 attention
   output `attn_out` and give Llama's FFN write a single `ffn_out` name before
   the residual add. The other trace nodes already have names. Lanternfish
   reads and writes them through the public scheduler callback and backend
   tensor APIs. No scheduler patch is necessary.
2. `0002-expose-cpu-ffn-gradient.patch`: Expose a read-only backward call for one
   dense Llama or Qwen2 FFN down weight. Use the native model graph and
   `ggml_build_backward_expand`. In training mode, size the common KV-path
   attention mask for the current ubatch instead of the padded cache. This
   lets the existing direct K/V path carry gradients through later attention
   layers.
3. `0003-add-cpu-residual-edit-backward.patch`: Differentiate the loss with
   respect to a residual displacement for the locate-then-edit methods.
4. `0004-accumulate-strided-f32-rows-on-gpu.patch`: Use the input row
   strides for `ACC`. The gradient of the query view can now stay on the
   GPU when the head and token dimensions are permuted.
5. `0005-respect-disabled-tf32-in-mmf.patch`: When
   `NVIDIA_TF32_OVERRIDE=0`, do not use the custom TF32 matrix kernel.
   cuBLAS already honors this setting.
6. `0006-admit-single-device-backward.patch`: Permit one fully offloaded
   CUDA or ROCm device for gradient and residual backward. Reject a CPU
   fallback for backward computation.
7. `0007-audit-complete-gpu-graphs.patch`: Check all active compute nodes
   before dispatch, including the forward prefix and ordinary edit decode.
   Lanternfish puts the input embeddings on the selected CUDA or ROCm device.
8. `0008-adapt-rdna3-bf16-wmma-operands.patch`: Preserve the BF16 operand
   bits for RDNA3 WMMA with both the `short` and `__bf16` compiler signatures.
9. `0009-select-weight-and-activation-gradients.patch`: Select a named weight
   or one activation row in the same strict graph. See the
   [selected gradient format](../docs/grad-select.md).
10. `0010-add-target-logit-objective.patch`: Differentiate a raw final-token
    logit with the same site, precision and placement checks. The loss API
    and its ABI do not change.

The HIP runtime uses this same series. The gfx1036 pointer-batched GEMM
workaround selects system-scope fences in the ROCm CLI process before
backend initialization. It does not change GEMM in llama.cpp. See the
[HIP build options and standalone reproducer](../docs/locate-edit.md#rocm-runtime).

Build Lanternfish with headers and shared libraries from the **same** patched
checkout. The gradient call adds a public symbol. An unpatched library cannot
link this version of Lanternfish. See [Install](../INSTALL.md#b2-llamacpp).
