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

Build Lanternfish with headers and shared libraries from the **same** patched
checkout. The gradient call adds a public symbol. An unpatched library cannot
link this version of Lanternfish. See [Install](../INSTALL.md#b2-llamacpp).
