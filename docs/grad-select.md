# Selected gradients, version 1

Use `grad-select` to differentiate one next-token loss or raw logit with respect to one
weight tensor or one activation row. The command does not edit the model.
It supports dense Llama and Qwen2 GGUF inputs with F32 and BF16 weights.
Use a new output directory for each call.

```sh
lanternfish tokenize --model model.gguf --text "The capital of France is"
lanternfish grad-select --model model.gguf --spec site.json \
  --backend cpu --threads 6 --out gradient-1
```

`tokenize` writes `{"tokens":[...]}` to stdout. It uses the model vocabulary
and special-token policy, without a chat template. `grad-select` accepts
these token IDs directly. It does not add BOS or EOS. Pass 1 to 256 IDs
within the model context. `target_id` selects a token after the input, not the
last input token. The default objective, `next_token_cross_entropy`, is
`-log softmax(final_logits)[target_id]`. Set `"objective":"target_logit"`
at the top level of the spec for the **raw, unnormalized** final-position
`final_logits[target_id]`. Both objectives accept all weight and activation sites.

Example `site.json` for a weight:

```json
{"tokens":[785,6722,315,9625,374],"target_id":12095,
 "site":{"kind":"weight","name":"blk.0.attn_q.weight"}}
```

`name` is the exact GGUF name, including `.weight` or `.bias`. The derivative
of a tied `token_embd.weight` includes its lookup and output projection uses.
RoPE frequency tables are constants, not trainable weights.

Example `site.json` for an activation:

```json
{"tokens":[785,6722,315,9625,374],"target_id":12095,
 "site":{"kind":"activation","role":"attn_out","layer":12,"position":1}}
```

`layer` and `position` start at zero. `attn_out` is the projected attention
write before the residual add. `ffn_out` is the FFN write before its residual
add. `l_out` is the block residual. These are the same sites as in
[`trace --intervention`](trace-format.md). Each result is the derivative
with respect to an additive displacement at the selected row, evaluated at
zero. A site with no effect on the final token can have a zero derivative.
For attribution patching from corrupt to clean, dot the corrupt-input
gradient with `activation_clean - activation_corrupt` at the same site.
With the default objective, this estimates the **change in loss**, not a change
in a logit. Negate it for a change in log probability. For X4, request
`target_logit` and dot the weight gradient with `quantized - reference` in
matching storage order. The result estimates the drift of that token's logit.
For a logit margin `logit[a] - logit[b]`, subtract the `target_logit` gradient
for `b` from the one for `a` at the same site and inputs. With cross entropy,
reverse the subtraction: `gradient(NLL_b) - gradient(NLL_a)`. The log
normalizers cancel.

## Output and checks

`gradient.f32` holds contiguous little-endian F32 values. `gradient.json`
is the completion record. Check `schema_version` (1), `objective`
(the requested objective), `dtype` (`f32_le`), `tokens`, `target_id`,
`site`, `backend`, and `placement` (`strict: true` and the device name).
Check the SHA-256 of the payload against `sha256`. The objective scalar
(`loss` for cross entropy; `logit` for `target_logit`) and all elements
must be finite. `shape` lists positive GGML extents from fastest to slowest;
the file size must be `4 * product(shape)`. The activation shape is
`[hidden_width]`; a matrix shape is `[input_width,output_width]`.
For a NumPy array use `fromfile(..., "<f4").reshape(shape[::-1])`.
To estimate empirical Fisher, square each sample gradient **before** averaging.

Use `--backend cpu`, `cuda` or `rocm`. GPU execution requires one supported,
fully offloaded device. Missing devices, failed placements, and unsupported
weights are errors. Use `--verbose` to record the scheduler audit for the
joint forward/backward graph. For a packaged runtime, verify `SHA256SUMS`
and set `LD_LIBRARY_PATH` to the matching `lib/<backend>` directory before
starting the CLI. The CPU tokenizer can use `lib/cpu`.

BF16 expansion needs temporary disk space for an F32 GGUF under the new
output directory. This file is unlinked after the private model load.
Budget host and device memory for the selected gradient, the graph and the
weights. The command does not split a request or use CPU as a GPU fallback.
