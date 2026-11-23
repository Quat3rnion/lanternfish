"""Generate deterministic two-layer GGUFs using only the standard library."""
import math
import struct


def write_model(path, precision, arch="llama", perturb=None, tied=False):
    def pack(fmt, *values):
        return struct.pack("<" + fmt, *values)

    def string(value):
        data = value.encode()
        return pack("Q", len(data)) + data

    vocab = ["<unk>", "<s>", "</s>", "▁", "a", "b", "c"]
    vocab += [f"<0x{i:02X}>" for i in range(256)]
    metadata = []

    def put(key, kind, data):
        metadata.append(string(key) + pack("I", kind) + data)

    put("general.architecture", 8, string(arch))
    for key, value in {
        "context_length": 128, "embedding_length": 32, "block_count": 2,
        "feed_forward_length": 64, "attention.head_count": 4,
        "attention.head_count_kv": 2, "rope.dimension_count": 8,
    }.items():
        put(arch + "." + key, 4, pack("I", value))
    moe = arch in ("qwen2moe", "qwen3moe")
    if moe:
        for key, value in {
            "expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": 32,
        }.items():
            put(f"{arch}.{key}", 4, pack("I", value))
    if arch == "qwen2moe":
        put(f"{arch}.expert_shared_feed_forward_length", 4, pack("I", 32))
    put(arch + ".attention.layer_norm_rms_epsilon", 6, pack("f", 1e-5))
    put("tokenizer.ggml.model", 8, string("llama"))
    put("tokenizer.ggml.tokens", 9, pack("IQ", 8, len(vocab)) + b"".join(map(string, vocab)))
    put("tokenizer.ggml.scores", 9, pack("IQ", 6, len(vocab)) + bytes(4 * len(vocab)))
    put("tokenizer.ggml.token_type", 9, pack("IQ", 5, len(vocab))
        + pack(f"{len(vocab)}i", 2, 3, 3, 1, 1, 1, 1, *([6] * 256)))
    for key, value in {"unknown": 0, "bos": 1, "eos": 2}.items():
        put(f"tokenizer.ggml.{key}_token_id", 4, pack("I", value))

    infos = []
    payload = bytearray()

    def tensor(name, width, rows=None, experts=None):
        kind = 2 if experts else ({"f32": 0, "bf16": 30, "q8_0": 8}[precision] if rows else 0)
        dims = [d for d in (width, rows, experts) if d is not None]
        infos.append(string(name) + pack("I", len(dims))
                     + pack("Q" * len(dims), *dims) + pack("IQ", kind, len(payload)))
        seed = len(infos)
        values = [0.10 * math.sin((i + 1) * (seed + 0.37)) for i in range(math.prod(dims))]
        if not rows:
            values = [1.0 + v for v in values]
        if perturb and name == perturb[0]:
            values[perturb[1]] += perturb[2]
        if kind == 0:
            payload.extend(pack("f" * len(values), *values))
        elif kind == 30:
            for value in values:
                bits = struct.unpack("<I", pack("f", value))[0]
                payload.extend(pack("H", (bits + 0x7fff + ((bits >> 16) & 1)) >> 16))
        elif kind == 8:
            for start in range(0, len(values), 32):
                block = values[start:start + 32]
                scale = max(map(abs, block)) / 127
                payload.extend(pack("e32b", scale, *(round(v / scale) for v in block)))
        else:
            for start in range(0, len(values), 32):
                block = values[start:start + 32]
                scale = max(map(abs, block)) / 7
                q = [round(v / scale) + 8 for v in block]
                payload.extend(pack("e16B", scale, *(a | b << 4 for a, b in zip(q[:16], q[16:]))))
        payload.extend(bytes(-len(payload) % 32))

    tensor("token_embd.weight", 32, len(vocab))
    if not tied:
        tensor("output.weight", 32, len(vocab))
    tensor("output_norm.weight", 32)
    for layer in range(2):
        prefix = f"blk.{layer}."
        for name, width, rows in [
            ("attn_norm", 32, None), ("attn_q", 32, 32), ("attn_k", 32, 16),
            ("attn_v", 32, 16), ("attn_output", 32, 32), ("ffn_norm", 32, None),
        ]:
            tensor(prefix + name + ".weight", width, rows)
        if moe:
            tensor(prefix + "ffn_gate_inp.weight", 32, 4)
            for name in ("gate", "up", "down"):
                tensor(prefix + f"ffn_{name}_exps.weight", 32, 32, 4)
            if arch == "qwen3moe":
                for name in ("q", "k"):
                    tensor(prefix + f"attn_{name}_norm.weight", 8)
            else:
                tensor(prefix + "ffn_gate_inp_shexp.weight", 32)
                for name in ("gate", "up", "down"):
                    tensor(prefix + f"ffn_{name}_shexp.weight", 32, 32)
        else:
            for name, width, rows in [("gate", 32, 64), ("up", 32, 64), ("down", 64, 32)]:
                tensor(prefix + f"ffn_{name}.weight", width, rows)
        if arch == "qwen2":
            for name, width in [("attn_q", 32), ("attn_k", 16), ("attn_v", 16)]:
                tensor(prefix + name + ".bias", width)
    if arch == "qwen2":
        tensor("output.bias", len(vocab))
    header = b"GGUF" + pack("IQQ", 3, len(infos), len(metadata))
    header += b"".join(metadata + infos)
    path.write_bytes(header + bytes(-len(header) % 32) + payload)
