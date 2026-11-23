//! Bounded, single-sequence CPU prompt capture.
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    capture, ffi,
    model::{ModelFile, Summary},
    viz::Lens,
};

const MAX_BATCH: usize = 32;
const MAX_TOKENS: usize = 4096;
pub(crate) const SCHEMA_VERSION: u32 = 1;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    intervention: Option<PathBuf>,
    #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = super::default_threads())]
    threads: i32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    LOut,
    AttnOut,
    FfnInp,
    FfnOut,
    KqSoftMax,
    FfnMoeLogits,
    FfnMoeTopk,
    FfnMoeRouteWeights,
}

impl Role {
    const ALL: [Self; 8] = [
        Self::LOut,
        Self::AttnOut,
        Self::FfnInp,
        Self::FfnOut,
        Self::KqSoftMax,
        Self::FfnMoeLogits,
        Self::FfnMoeTopk,
        Self::FfnMoeRouteWeights,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::LOut => "l_out",
            Self::AttnOut => "attn_out",
            Self::FfnInp => "ffn_inp",
            Self::FfnOut => "ffn_out",
            Self::KqSoftMax => "kq_soft_max",
            Self::FfnMoeLogits => "ffn_moe_logits",
            Self::FfnMoeTopk => "ffn_moe_topk",
            Self::FfnMoeRouteWeights => "ffn_moe_route_weights",
        }
    }

    fn routing(self) -> bool {
        matches!(
            self,
            Self::FfnMoeLogits | Self::FfnMoeTopk | Self::FfnMoeRouteWeights
        )
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Spec {
    pub(crate) layers: BTreeSet<usize>,
    pub(crate) roles: BTreeSet<Role>,
    pub(crate) positions: BTreeSet<usize>,
    pub(crate) top_k: usize,
    pub(crate) byte_cap: usize,
}

#[derive(Deserialize, Serialize)]
struct Rule {
    layer: usize,
    role: Role,
    positions: BTreeSet<usize>,
    #[serde(flatten)]
    operation: Operation,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Operation {
    Add { vector: Vec<f32> },
    Replace { vector: Vec<f32> },
    ProjectOut { vector: Vec<f32> },
    Subtract { vector: Vec<f32>, scale: f32 },
    Scale { factor: f32 },
    Zero,
    ZeroExpert { expert: usize },
}

impl Operation {
    fn validate(&self, hidden: usize) -> Result<()> {
        match self {
            Self::Add { vector }
            | Self::Replace { vector }
            | Self::ProjectOut { vector }
            | Self::Subtract { vector, .. } => ensure!(
                vector.len() == hidden && vector.iter().all(|v| v.is_finite()),
                "intervention vector must contain {hidden} finite values"
            ),
            Self::Scale { factor } => ensure!(factor.is_finite(), "scale must be finite"),
            Self::Zero | Self::ZeroExpert { .. } => {}
        }
        match self {
            Self::ProjectOut { vector } => ensure!(
                vector.iter().any(|&v| v != 0.0),
                "projection direction must be nonzero"
            ),
            Self::Subtract { scale, .. } => {
                ensure!(
                    scale.is_finite() && *scale >= 0.0,
                    "subtraction scale must be finite and nonnegative"
                )
            }
            _ => {}
        }
        Ok(())
    }

    fn apply(&self, row: &mut [f32]) {
        match self {
            Self::Add { vector } => row.iter_mut().zip(vector).for_each(|(a, b)| *a += b),
            Self::Replace { vector } => row.copy_from_slice(vector),
            Self::ProjectOut { vector } => {
                // F64 keeps the norm squared representable for finite F32 directions.
                let (dot, norm_squared) =
                    row.iter()
                        .zip(vector)
                        .fold((0.0, 0.0), |(dot, norm_squared), (&x, &d)| {
                            let d = f64::from(d);
                            (dot + f64::from(x) * d, norm_squared + d * d)
                        });
                let coefficient = dot / norm_squared;
                for (x, &d) in row.iter_mut().zip(vector) {
                    *x = (f64::from(*x) - coefficient * f64::from(d)) as f32;
                }
            }
            Self::Subtract { vector, scale } => {
                for (x, &d) in row.iter_mut().zip(vector) {
                    *x -= *scale * d;
                }
            }
            Self::Scale { factor } => row.iter_mut().for_each(|a| *a *= factor),
            Self::Zero | Self::ZeroExpert { .. } => row.fill(0.0),
        }
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize)]
pub(crate) struct Key {
    pub(crate) layer: usize,
    pub(crate) role: Role,
    pub(crate) position: usize,
}

pub(crate) struct Row {
    shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
}

struct Eval<'a> {
    spec: &'a Spec,
    hidden: usize,
    first: usize,
    count: usize,
    rows: BTreeMap<Key, Row>,
    pending: BTreeMap<Key, &'a Operation>,
    edited: BTreeMap<Key, Option<usize>>,
    expert_ids: BTreeMap<Key, Vec<i32>>,
    lens: bool,
    bytes: usize,
    error: Option<anyhow::Error>,
}

impl Eval<'_> {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .context("capture byte size overflows")?;
        ensure!(self.bytes <= self.spec.byte_cap, "capture exceeds byte_cap");
        Ok(())
    }

    fn selected(&self, key: Key) -> bool {
        self.spec.layers.contains(&key.layer)
            && self.spec.positions.contains(&key.position)
            && (self.spec.roles.contains(&key.role) || (self.lens && key.role == Role::LOut))
    }

    fn targets(&self, role: Role, layer: usize) -> impl Iterator<Item = Key> + use<> {
        (self.first..self.first + self.count).map(move |position| Key {
            layer,
            role,
            position,
        })
    }

    fn read_node(&mut self, node: *mut ffi::ggml_tensor, role: Role, layer: usize) -> Result<()> {
        if role.routing() {
            return self.read_route(node, role, layer);
        }
        let tensor = unsafe { &*node };
        let attention = role == Role::KqSoftMax;
        let [width, rows, heads, seq] = tensor.ne.map(|n| n as usize);
        ensure!(
            tensor.type_ == ffi::ggml_type::GGML_TYPE_F32
                && tensor.nb[0] == 4
                && width != 0
                && rows == self.count
                && seq == 1
                && (attention || (heads == 1 && width == self.hidden)),
            "unsupported shape/type at {}-{layer}: {:?} {:?}",
            role.name(),
            tensor.ne,
            tensor.type_
        );
        let shape = if attention {
            vec![heads, width]
        } else {
            vec![width]
        };
        let len = width
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(4))
            .context("tensor byte size overflows")?;
        for key in self.targets(role, layer) {
            let capture = self.selected(key);
            let op = if self.expert_edit_at(key) && role == Role::FfnOut {
                None
            } else {
                self.pending.remove(&key)
            };
            if !(capture || op.is_some()) {
                continue;
            }
            if capture {
                ensure!(
                    !self.rows.contains_key(&key),
                    "duplicate graph node at {key:?}"
                );
                if self.spec.roles.contains(&key.role) {
                    self.charge(len)?;
                }
            }
            let offset = (key.position - self.first) * tensor.nb[1];
            let head_stride = tensor.nb[2];
            let mut values = vec![0.0f32; len / 4];
            for (head, row) in values.chunks_exact_mut(width).enumerate() {
                unsafe {
                    ffi::ggml_backend_tensor_get(
                        node,
                        row.as_mut_ptr().cast(),
                        offset + head * head_stride,
                        width * 4,
                    );
                }
            }
            if let Some(op) = op {
                ensure!(tensor.view_src.is_null(), "cannot edit a view at {key:?}");
                op.apply(&mut values);
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite intervention at {key:?}"
                );
                unsafe { ffi::ggml_backend_tensor_set(node, values.as_ptr().cast(), offset, len) };
                self.edited.insert(key, None);
            }
            if capture {
                self.rows.insert(
                    key,
                    Row {
                        shape: shape.clone(),
                        values,
                    },
                );
            }
        }
        Ok(())
    }

    fn expert_edit_at(&self, key: Key) -> bool {
        matches!(
            self.pending.get(&Key {
                role: Role::FfnOut,
                ..key
            }),
            Some(Operation::ZeroExpert { .. })
        )
    }

    fn read_route(&mut self, node: *mut ffi::ggml_tensor, role: Role, layer: usize) -> Result<()> {
        let tensor = unsafe { &*node };
        let [n0, n1, n2, n3] = tensor.ne.map(|n| n as usize);
        let ids = role == Role::FfnMoeTopk;
        let weights = role == Role::FfnMoeRouteWeights;
        let (width, count, stride) = if weights {
            ensure!(
                n0 == 1 && tensor.nb[1] == 4,
                "unsupported route weight layout"
            );
            (n1, n2, tensor.nb[2])
        } else {
            ensure!(n2 == 1, "unsupported router row layout");
            (n0, n1, tensor.nb[1])
        };
        ensure!(
            tensor.type_
                == if ids {
                    ffi::ggml_type::GGML_TYPE_I32
                } else {
                    ffi::ggml_type::GGML_TYPE_F32
                }
                && tensor.nb[0] == 4
                && width != 0
                && count == self.count
                && n3 == 1,
            "unsupported shape/type at {}-{layer}: {:?} {:?}",
            role.name(),
            tensor.ne,
            tensor.type_
        );
        for key in self.targets(role, layer) {
            let capture = self.selected(key);
            let edit = ids && self.expert_edit_at(key);
            if !capture && !edit {
                continue;
            }
            let len = width.checked_mul(4).context("router row size overflows")?;
            if capture {
                self.charge(len)?;
            }
            let mut bits = vec![0i32; width];
            unsafe {
                ffi::ggml_backend_tensor_get(
                    node,
                    bits.as_mut_ptr().cast(),
                    (key.position - self.first) * stride,
                    len,
                );
            }
            if capture {
                // The v1 payload type is f32_le. Small GGUF expert IDs are exact in f32.
                ensure!(
                    !ids || bits.iter().all(|&id| (0..=16_777_216).contains(&id)),
                    "expert ID is outside the exact f32 range"
                );
                let values = bits
                    .iter()
                    .map(|&v| {
                        if ids {
                            v as f32
                        } else {
                            f32::from_bits(v as u32)
                        }
                    })
                    .collect();
                ensure!(
                    self.rows
                        .insert(
                            key,
                            Row {
                                shape: vec![width],
                                values
                            }
                        )
                        .is_none(),
                    "duplicate graph node at {key:?}"
                );
            }
            if edit {
                self.expert_ids.insert(key, bits);
            }
        }
        Ok(())
    }

    fn zero_experts(&mut self, node: *mut ffi::ggml_tensor, layer: usize) -> Result<()> {
        let tensor = unsafe { &*node };
        let [hidden, slots, count, seq] = tensor.ne.map(|n| n as usize);
        ensure!(
            tensor.type_ == ffi::ggml_type::GGML_TYPE_F32
                && tensor.view_src.is_null()
                && hidden == self.hidden
                && count == self.count
                && seq == 1
                && tensor.nb[0] == 4,
            "unsupported weighted expert contribution layout"
        );
        let zeros = vec![0.0f32; hidden];
        for key in self.targets(Role::FfnOut, layer) {
            if !self.expert_edit_at(key) {
                continue;
            }
            let Some(Operation::ZeroExpert { expert }) = self.pending.remove(&key) else {
                unreachable!();
            };
            let ids = self
                .expert_ids
                .remove(&Key {
                    role: Role::FfnMoeTopk,
                    ..key
                })
                .context("missing expert IDs for ablation")?;
            ensure!(
                ids.len() == slots,
                "expert ID and weight slot counts differ"
            );
            if let Some(slot) = ids.iter().position(|&id| id as usize == *expert) {
                let offset = (key.position - self.first) * tensor.nb[2] + slot * tensor.nb[1];
                unsafe {
                    ffi::ggml_backend_tensor_set(node, zeros.as_ptr().cast(), offset, hidden * 4);
                }
            }
            // A valid but unselected expert is a no-op.
            self.edited.insert(key, Some(*expert));
        }
        Ok(())
    }
}

unsafe extern "C" fn eval_callback(
    node: *mut ffi::ggml_tensor,
    ask: bool,
    data: *mut c_void,
) -> bool {
    let eval = unsafe { &mut *data.cast::<Eval<'_>>() };
    let name = unsafe { CStr::from_ptr((*node).name.as_ptr()) }.to_string_lossy();
    let Some((base, layer)) = capture::split_name(&name) else {
        return !ask;
    };
    if eval.error.is_some() {
        return !ask;
    }
    if base == "ffn_moe_weighted" {
        if ask {
            return eval
                .targets(Role::FfnOut, layer)
                .any(|key| eval.expert_edit_at(key));
        }
        eval.error = eval.zero_experts(node, layer).err();
        return true;
    }
    let Some(role) = Role::ALL.into_iter().find(|r| r.name() == base) else {
        return !ask;
    };
    if ask {
        return eval.targets(role, layer).any(|key| {
            eval.selected(key)
                || (role == Role::FfnMoeTopk && eval.expert_edit_at(key))
                || (eval.pending.contains_key(&key) && !eval.expert_edit_at(key))
        });
    }
    eval.error = eval.read_node(node, role, layer).err();
    true
}

#[derive(Serialize)]
struct Logit {
    token_id: usize,
    logit: f32,
}

#[derive(Serialize)]
struct Logits {
    position: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer: Option<usize>,
    top: Vec<Logit>,
}

fn top(logits: &[f32], k: usize) -> Result<Vec<Logit>> {
    ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    let cmp = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
    if k < ids.len() {
        ids.select_nth_unstable_by(k, cmp);
        ids.truncate(k);
    }
    ids.sort_unstable_by(cmp);
    Ok(ids
        .into_iter()
        .map(|token_id| Logit {
            token_id,
            logit: logits[token_id],
        })
        .collect())
}

pub(crate) fn check_threads(threads: i32) -> Result<()> {
    ensure!(threads > 0, "threads must be positive");
    Ok(())
}

struct Forward {
    tokens: Vec<i32>,
    rows: BTreeMap<Key, Row>,
    logits: Vec<Logits>,
    lenses: Vec<Logits>,
    edited: BTreeMap<Key, Option<usize>>,
}

fn emit_logits(
    eval: &mut Eval<'_>,
    out: &mut Vec<Logits>,
    position: usize,
    layer: Option<usize>,
    values: &[f32],
) -> Result<()> {
    let row = Logits {
        position,
        layer,
        top: top(values, eval.spec.top_k)?,
    };
    eval.charge(serde_json::to_vec(&row)?.len())?;
    out.push(row);
    Ok(())
}

struct Batch(ffi::llama_batch);
impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { ffi::llama_batch_free(self.0) }
    }
}

fn forward(
    model: &ModelFile,
    prompt: &str,
    spec: &Spec,
    rules: &[Rule],
    threads: i32,
) -> Result<Forward> {
    check_threads(threads)?;
    ensure!(spec.byte_cap > 0, "byte_cap must be positive");
    ensure!(!spec.positions.is_empty(), "positions must not be empty");
    ensure!(
        spec.layers.iter().all(|n| *n < model.s.n_layers),
        "capture layer outside model"
    );
    ensure!(
        !spec.layers.is_empty() || spec.roles.is_empty(),
        "roles require layers"
    );
    let moe = matches!(model.s.arch.as_str(), "qwen2moe" | "qwen3moe");
    ensure!(
        moe || !spec.roles.iter().any(|role| role.routing()),
        "routing capture supports only Qwen2MoE and Qwen3MoE"
    );
    let use_lens = spec.top_k > 0 && !spec.layers.is_empty() && !moe;
    let raw = capture::load_model(&model.s.path, true)?;
    let mut eval = Eval {
        spec,
        hidden: model.s.hidden,
        first: 0,
        count: 0,
        rows: BTreeMap::new(),
        pending: BTreeMap::new(),
        edited: BTreeMap::new(),
        expert_ids: BTreeMap::new(),
        lens: use_lens,
        bytes: 0,
        error: None,
    };
    let mut ll = capture::Llama {
        model: raw,
        ctx: ptr::null_mut(),
    };
    let vocab = unsafe { ffi::llama_model_get_vocab(raw) };
    let tokens = capture::tokenize(vocab, prompt)?;
    let train_ctx = unsafe { ffi::llama_model_n_ctx_train(raw) } as usize;
    ensure!(
        !tokens.is_empty() && tokens.len() <= MAX_TOKENS.min(train_ctx),
        "prompt must contain 1..={} tokens",
        MAX_TOKENS.min(train_ctx)
    );
    let nv = unsafe { ffi::llama_vocab_n_tokens(vocab) } as usize;
    ensure!(spec.top_k <= nv, "top_k exceeds vocabulary size");
    ensure!(
        spec.positions.iter().all(|p| *p < tokens.len()),
        "capture position outside prompt"
    );
    // Every requested activation produces at least a tensor entry or a missing
    // entry. Every requested logit has at least one byte in the manifest.
    let entries_min = spec
        .layers
        .len()
        .saturating_mul(spec.roles.len())
        .max(spec.top_k)
        .saturating_mul(spec.positions.len());
    ensure!(entries_min <= spec.byte_cap, "export exceeds byte_cap");
    let lens = if use_lens {
        ensure!(
            matches!(model.s.arch.as_str(), "llama" | "qwen2"),
            "logit lens supports only Llama and Qwen2; omit layers or set top_k to 0"
        );
        let lens = Lens::load(model)?;
        let bias = if model.s.arch == "qwen2" && model.g.has("output.bias") {
            Some(model.g.rows("output.bias", lens.vocab, 0, 1)?)
        } else {
            None
        };
        Some((lens, rms_epsilon(model)?, bias))
    } else {
        None
    };
    for rule in rules {
        ensure!(
            rule.layer < model.s.n_layers,
            "intervention layer outside model"
        );
        match rule.operation {
            Operation::ZeroExpert { expert } => ensure!(
                moe && rule.role == Role::FfnOut && expert < model.s.n_experts,
                "zero_expert requires a valid Qwen2MoE or Qwen3MoE expert and ffn_out"
            ),
            _ => ensure!(
                matches!(model.s.arch.as_str(), "llama" | "qwen2")
                    && rule.role != Role::KqSoftMax
                    && !rule.role.routing(),
                "activation interventions require a writable Llama or Qwen2 role"
            ),
        };
        ensure!(
            !rule.positions.is_empty(),
            "intervention positions must not be empty"
        );
        rule.operation.validate(model.s.hidden)?;
        for &position in &rule.positions {
            ensure!(
                position < tokens.len(),
                "intervention position outside prompt"
            );
            let key = Key {
                layer: rule.layer,
                role: rule.role,
                position,
            };
            ensure!(
                eval.pending.insert(key, &rule.operation).is_none(),
                "overlapping interventions at {key:?}"
            );
        }
    }
    // Every row is a logit row. This keeps the final layer's query rows aligned
    // with the batch instead of pruning them to the last token.
    let n_batch = tokens.len().min(MAX_BATCH);
    let mut params = capture::cpu_context_params(tokens.len(), n_batch, threads);
    params.cb_eval = Some(eval_callback);
    params.cb_eval_user_data = (&raw mut eval).cast();
    ll.ctx = unsafe { ffi::llama_init_from_model(raw, params) };
    ensure!(!ll.ctx.is_null(), "cannot create trace context");
    let mut batch = Batch(unsafe { ffi::llama_batch_init(n_batch as i32, 0, 1) });
    let mut logits = Vec::new();
    let mut lenses = Vec::new();
    for (chunk, input) in tokens.chunks(n_batch).enumerate() {
        let first = chunk * n_batch;
        eval.first = first;
        eval.count = input.len();
        batch.0.n_tokens = input.len() as i32;
        for (i, &token) in input.iter().enumerate() {
            unsafe {
                *batch.0.token.add(i) = token;
                *batch.0.pos.add(i) = (first + i) as i32;
                *batch.0.n_seq_id.add(i) = 1;
                **batch.0.seq_id.add(i) = 0;
                *batch.0.logits.add(i) = 1;
            }
        }
        let status = unsafe { ffi::llama_decode(ll.ctx, batch.0) };
        unsafe { ffi::llama_synchronize(ll.ctx) };
        if let Some(error) = eval.error.take() {
            return Err(error);
        }
        ensure!(status == 0, "llama_decode failed: {status}");
        if spec.top_k > 0 {
            for &position in spec.positions.range(first..first + input.len()) {
                let vals = unsafe { ffi::llama_get_logits_ith(ll.ctx, (position - first) as i32) };
                ensure!(!vals.is_null(), "missing logits at position {position}");
                let vals = unsafe { std::slice::from_raw_parts(vals, nv) };
                emit_logits(&mut eval, &mut logits, position, None, vals)?;
            }
        }
        if let Some((lens, eps, bias)) = &lens {
            for &layer in &spec.layers {
                for &position in spec.positions.range(first..first + input.len()) {
                    let key = Key {
                        layer,
                        role: Role::LOut,
                        position,
                    };
                    if let Some(row) = eval.rows.get(&key) {
                        let mut vals = lens.rms_logits(&[&row.values], *eps);
                        if let Some(bias) = bias {
                            vals.iter_mut().zip(bias).for_each(|(v, b)| *v += b);
                        }
                        emit_logits(&mut eval, &mut lenses, position, Some(layer), &vals)?;
                        if !spec.roles.contains(&Role::LOut) {
                            eval.rows.remove(&key);
                        }
                    }
                }
            }
        }
    }
    drop(ll);
    if let Some(key) = eval.pending.keys().next() {
        bail!("intervention node not found at {key:?}");
    }
    lenses.sort_by_key(|l| (l.layer, l.position));
    Ok(Forward {
        tokens,
        rows: eval.rows,
        logits,
        lenses,
        edited: eval.edited,
    })
}

pub(crate) fn rms_epsilon(model: &ModelFile) -> Result<f32> {
    model
        .g
        .float(&format!(
            "{}.attention.layer_norm_rms_epsilon",
            model.s.arch
        ))
        .filter(|v| v.is_finite() && *v > 0.0)
        .context("missing or invalid RMS norm epsilon")
}

#[derive(Serialize)]
pub(crate) struct Payload {
    file: String,
    shape: Vec<usize>,
    dtype: &'static str,
    bytes: usize,
    sha256: String,
}

impl Payload {
    pub(crate) fn f32(file: String, shape: Vec<usize>, values: &[f32]) -> (Self, Vec<u8>) {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sha256 = Sha256::digest(&data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        (
            Self {
                file,
                shape,
                dtype: "f32_le",
                bytes: data.len(),
                sha256,
            },
            data,
        )
    }
}

pub(crate) fn write_export(
    out: &Path,
    manifest: &[u8],
    files: &[(Payload, Vec<u8>)],
) -> Result<()> {
    fs::create_dir(out)
        .with_context(|| format!("create new output directory {}", out.display()))?;
    for (payload, data) in files {
        fs::write(out.join(&payload.file), data)?;
    }
    // Publish the manifest only after all payloads are complete.
    let pending = out.join("manifest.json.part");
    fs::write(&pending, manifest)?;
    fs::rename(pending, out.join("manifest.json"))?;
    Ok(())
}

#[derive(Serialize)]
struct Tensor<'a> {
    #[serde(flatten)]
    key: Key,
    node: String,
    #[serde(flatten)]
    payload: &'a Payload,
}

#[derive(Serialize)]
struct Token {
    position: usize,
    token_id: i32,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u32,
    kind: &'static str,
    model: &'a Summary,
    spec: &'a Spec,
    tokens: Vec<Token>,
    tensors: Vec<Tensor<'a>>,
    missing: Vec<Key>,
    interventions: Vec<Applied>,
    model_logits: Vec<Logits>,
    logit_lens: Vec<Logits>,
}

#[derive(Serialize)]
struct Applied {
    #[serde(flatten)]
    key: Key,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert: Option<usize>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let spec: Spec =
        serde_json::from_slice(&fs::read(&args.spec)?).context("parse capture spec")?;
    let rules: Vec<Rule> = if let Some(path) = args.intervention {
        serde_json::from_slice(&fs::read(path)?).context("parse intervention list")?
    } else {
        Vec::new()
    };
    let model = ModelFile::open(&args.model)?;
    let captured = forward(&model, &args.prompt, &spec, &rules, args.threads)?;
    let mut keys = Vec::new();
    let mut files = Vec::new();
    let mut missing = Vec::new();
    let mut rows = captured.rows;
    for &layer in &spec.layers {
        for &role in &spec.roles {
            for &position in &spec.positions {
                let key = Key {
                    layer,
                    role,
                    position,
                };
                if let Some(row) = rows.remove(&key) {
                    let file = format!("tensor-{:06}.f32", files.len());
                    files.push(Payload::f32(file, row.shape, &row.values));
                    keys.push(key);
                } else {
                    missing.push(key);
                }
            }
        }
    }
    let tensors: Vec<_> = keys
        .iter()
        .zip(&files)
        .map(|(&key, (payload, _))| Tensor {
            key,
            node: format!("{}-{}", key.role.name(), key.layer),
            payload,
        })
        .collect();
    let manifest = serde_json::to_vec(&Manifest {
        schema_version: SCHEMA_VERSION,
        kind: "trace",
        model: &model.s,
        spec: &spec,
        tokens: captured
            .tokens
            .into_iter()
            .enumerate()
            .map(|(position, token_id)| Token { position, token_id })
            .collect(),
        tensors,
        missing,
        interventions: captured
            .edited
            .into_iter()
            .map(|(key, expert)| Applied { key, expert })
            .collect(),
        model_logits: captured.logits,
        logit_lens: captured.lenses,
    })?;
    let bytes = files.iter().try_fold(manifest.len(), |total, (_, data)| {
        total
            .checked_add(data.len())
            .context("export size overflows")
    })?;
    ensure!(bytes <= spec.byte_cap, "export exceeds byte_cap");
    write_export(&args.out, &manifest, &files)
}
