# Selected gradients, version 1

Use `grad-select` to differentiate one next-token loss with respect to one
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
within the model context. `target_id` is the observed next token, not the
last input token. The scalar objective is `-log softmax(final_logits)[target_id]`.

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
This estimates the **change in loss**, not a change in a logit. Negate it
for a change in log probability. Do not use it as a raw logit derivative.
For the logit margin `logit[a] - logit[b]`, use `gradient(target_id=b) -
gradient(target_id=a)` with identical inputs and site. The log normalizers
cancel exactly. This gives X4 and X5 a contrast in logit units without a
second type of objective.

## Output and checks

`gradient.f32` holds contiguous little-endian F32 values. `gradient.json`
is the completion record. Check `schema_version` (1), `objective`
(`next_token_cross_entropy`), `dtype` (`f32_le`), `tokens`, `target_id`,
`site`, `backend`, and `placement` (`strict: true` and the device name).
Check the SHA-256 of the payload against `sha256`. `loss` and all elements
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
