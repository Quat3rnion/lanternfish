//! Locate-then-edit against a private model copy. Emits factors, not weights.
use std::ffi::{CStr, CString, c_void};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    backward_backend::{Backend, Device},
    capture, edit_solve, ffi,
    model::ModelFile,
    trace,
};

const SAMPLE: &str = include_str!("../data/edit-moment.txt");
const SAMPLE_SHA: &str = "21ffd89a9f728b1b760433afd385852589cf5012aac10336ac33b2ad23609877";
const MAX_TOKENS: usize = 128;
const MAX_TARGET: usize = 12;
const MAX_CACHED_MOMENTS: usize = 12;
const PREFIXES: [&str; 2] = ["", "Here is a note. "];
const CONTEXT_GROUPS: [usize; 2] = [1, 1];

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    request: PathBuf,
    #[arg(long)]
    cache: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Backend for the model graph. CUDA backward does not fall back to CPU.
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,
    #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = super::default_threads())]
    threads: i32,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Method {
    Rome,
    Memit,
    Alphaedit,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fact {
    subject: String,
    prompt: String,
    target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    method: Method,
    layers: Vec<usize>,
    facts: Vec<Fact>,
    #[serde(default = "default_steps")]
    steps: usize,
}

fn default_steps() -> usize {
    16
}

#[derive(Serialize)]
struct Update {
    layer: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

struct ContextPtr(*mut ffi::llama_context);
impl Drop for ContextPtr {
    fn drop(&mut self) {
        unsafe { ffi::llama_free(self.0) };
    }
}

#[derive(Default)]
struct Readout {
    layer: usize,
    z_layer: usize,
    keys: Vec<Vec<f32>>,
    residual: Vec<Vec<f32>>,
    mlp: Vec<Vec<f32>>,
}

unsafe extern "C" fn collect(t: *mut ffi::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    let c = unsafe { &mut *ud.cast::<Readout>() };
    let t = unsafe { &*t };
    let name = unsafe { CStr::from_ptr(t.name.as_ptr()) }.to_str();
    let Some((name, layer)) = name.ok().and_then(capture::split_name) else {
        return !ask;
    };
    let rows = match name {
        "ffn_key" if layer == c.layer => &mut c.keys,
        "l_out" if layer == c.z_layer => &mut c.residual,
        "ffn_out" if layer == c.z_layer => &mut c.mlp,
        _ => return !ask,
    };
    if !ask && t.type_ == ffi::ggml_type::GGML_TYPE_F32 && t.nb[0] == 4 {
        *rows = (0..t.ne[1] as usize)
            .map(|i| {
                let mut row = vec![0.0; t.ne[0] as usize];
                unsafe {
                    ffi::ggml_backend_tensor_get(
                        t,
                        row.as_mut_ptr().cast(),
                        i * t.nb[1],
                        row.len() * 4,
                    );
                }
                row
            })
            .collect();
    }
    true
}

struct Engine {
    ll: capture::Llama,
    device: Device,
    threads: i32,
    n_ctx: usize,
    vocab: *const ffi::llama_vocab,
    nv: usize,
    cols: usize,
    rows: usize,
    bf16: bool,
}

impl Engine {
    fn context(&self, n: usize, state: Option<&mut Readout>) -> Result<ContextPtr> {
        ensure!(
            (1..=self.n_ctx).contains(&n),
            "prompt exceeds {} tokens",
            self.n_ctx
        );
        let mut params = self.device.context_params(n, self.threads);
        if let Some(state) = state {
            params.cb_eval = Some(collect);
            params.cb_eval_user_data = (state as *mut Readout).cast();
        }
        let ptr = unsafe { ffi::llama_init_from_model(self.ll.model, params) };
        ensure!(
            !ptr.is_null(),
            "cannot create edit context on {}",
            self.device.name()
        );
        Ok(ContextPtr(ptr))
    }

    fn forward(&self, tokens: &[i32], state: Option<&mut Readout>) -> Result<Vec<f32>> {
        let ctx = self.context(tokens.len(), state)?;
        let mut input = tokens.to_owned();
        let mut output = vec![1; tokens.len()];
        let mut batch = unsafe { ffi::llama_batch_get_one(input.as_mut_ptr(), input.len() as i32) };
        batch.logits = output.as_mut_ptr();
        ensure!(
            unsafe { ffi::llama_decode(ctx.0, batch) } == 0,
            "edit forward failed"
        );
        let ptr = unsafe { ffi::llama_get_logits_ith(ctx.0, -1) };
        ensure!(!ptr.is_null(), "missing edit logits");
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.nv) }.to_vec())
    }

    fn sample(&self, tokens: &[i32], layer: usize, z_layer: usize) -> Result<Readout> {
        let mut c = Readout {
            layer,
            z_layer,
            ..Default::default()
        };
        self.forward(tokens, Some(&mut c))?;
        ensure!(
            c.keys.len() == tokens.len()
                && c.residual.len() == tokens.len()
                && c.mlp.len() == tokens.len()
                && c.keys.iter().all(|v| v.len() == self.cols)
                && c.residual
                    .iter()
                    .chain(&c.mlp)
                    .all(|v| v.len() == self.rows),
            "missing or invalid FFN key or residual"
        );
        Ok(c)
    }

    fn grad(&self, example: &Example, layer: usize, delta: &[f32], g: &mut [f32]) -> Result<f32> {
        let ctx = self.context(example.tokens.len(), None)?;
        let mut loss = 0.0;
        let ret = unsafe {
            ffi::llama_lanternfish_residual_gradient(
                ctx.0,
                example.tokens.as_ptr(),
                example.tokens.len() as i32,
                example.first as i32,
                example.labels.as_ptr(),
                layer as i32,
                example.pos as i32,
                delta.as_ptr(),
                &mut loss,
                g.as_mut_ptr(),
                g.len(),
            )
        };
        ensure!(
            ret == 0,
            "residual backward failed on {} (status {ret}); use --verbose for the llama.cpp error",
            self.device.name()
        );
        ensure!(
            loss.is_finite() && g.iter().all(|v| v.is_finite()),
            "non-finite residual loss or gradient"
        );
        Ok(loss)
    }

    fn tokenize(&self, s: &str, special: bool) -> Result<Vec<i32>> {
        if special {
            return capture::tokenize(self.vocab, s);
        }
        let mut buf = vec![0; s.len() + 16];
        let n = unsafe {
            ffi::llama_tokenize(
                self.vocab,
                s.as_ptr().cast(),
                s.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                false,
                false,
            )
        };
        ensure!(n > 0, "empty or invalid target");
        buf.truncate(n as usize);
        Ok(buf)
    }

    fn locate(&self, prompt: &str, subject: &str, prefix: &str) -> Result<(Vec<i32>, usize)> {
        let (before, after) = prompt
            .split_once("{}")
            .context("prompt needs a {} subject slot")?;
        let partial = format!("{prefix}{before}{subject}");
        let tokens = self.tokenize(&format!("{partial}{after}"), true)?;
        let partial = self.tokenize(&partial, true)?;
        let n = partial
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count();
        // If BPE merges the final subject characters with the suffix, choose
        // the token at the first mismatch rather than the preceding token.
        let pos = if n == partial.len() {
            n.saturating_sub(1)
        } else {
            n
        };
        ensure!(
            n > 0 && pos < tokens.len(),
            "cannot locate final subject token"
        );
        Ok((tokens, pos))
    }

    fn apply(&self, update: &Update) -> Result<()> {
        let t = unsafe { ffi::llama_lanternfish_ffn_down(self.ll.model, update.layer as i32) };
        ensure!(!t.is_null(), "missing FFN down");
        let shape = unsafe { &*t };
        ensure!(
            shape.type_ == ffi::ggml_type::GGML_TYPE_F32
                && shape.ne == [self.cols as i64, self.rows as i64, 1, 1],
            "wrong FFN down shape or type"
        );
        let mut weight = vec![0.0f32; self.rows * self.cols];
        unsafe { ffi::ggml_backend_tensor_get(t, weight.as_mut_ptr().cast(), 0, weight.len() * 4) };
        for (r, row) in weight.chunks_exact_mut(self.cols).enumerate() {
            for (c, w) in row.iter_mut().enumerate() {
                // Same F32 accumulation order as the W3 factor operation.
                let delta = update
                    .keys
                    .iter()
                    .zip(&update.values)
                    .fold(0.0, |acc, (k, v)| acc + v[r] * k[c]);
                *w += delta;
                if self.bf16 {
                    let b = w.to_bits();
                    *w = f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000);
                }
                ensure!(w.is_finite(), "non-finite updated weight");
            }
        }
        unsafe { ffi::ggml_backend_tensor_set(t, weight.as_ptr().cast(), 0, weight.len() * 4) };
        Ok(())
    }
}

struct Example {
    tokens: Vec<i32>,
    pos: usize,
    first: usize,
    labels: Vec<f32>,
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn softmax(a: &mut [f32]) {
    let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    a.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = a.iter().sum();
    a.iter_mut().for_each(|x| *x /= sum);
}

fn optimize(
    e: &Engine,
    fact: &Fact,
    method: Method,
    layer: usize,
    steps: usize,
) -> Result<(Vec<f32>, [f32; 2])> {
    let target = if fact.target.starts_with(char::is_whitespace) {
        fact.target.clone()
    } else {
        format!(" {}", fact.target)
    };
    let target = e.tokenize(&target, false)?;
    ensure!(target.len() <= MAX_TARGET, "too many target tokens");
    let (original, pos) = e.locate(&fact.prompt, &fact.subject, "")?;
    let initial = e.sample(&original, layer, layer)?;
    let ref_value = if method == Method::Rome {
        &initial.mlp[pos]
    } else {
        &initial.residual[pos]
    };
    let ref_norm = norm(ref_value).max(1e-4);
    let mut rewrite = Vec::new();
    for prefix in PREFIXES {
        let (mut tokens, pos) = e.locate(&fact.prompt, &fact.subject, prefix)?;
        let first = tokens.len() - 1;
        tokens.extend_from_slice(&target[..target.len() - 1]);
        let mut labels = vec![0.0; target.len() * e.nv];
        for (row, id) in target.iter().enumerate() {
            labels[row * e.nv + *id as usize] = 1.0;
        }
        rewrite.push(Example {
            tokens,
            pos,
            first,
            labels,
        });
    }
    let (mut tokens, pos) = e.locate("{} is a", &fact.subject, "")?;
    tokens.truncate(pos + 1);
    let mut labels = e.forward(&tokens, None)?;
    softmax(&mut labels);
    let kl = Example {
        first: pos,
        pos,
        tokens,
        labels,
    };

    let mut delta = vec![0.0; e.rows];
    let mut best = delta.clone();
    let mut m = delta.clone();
    let mut v = delta.clone();
    let mut g = delta.clone();
    let mut temp = delta.clone();
    let mut initial_loss = 0.0;
    let mut best_loss = f32::INFINITY;
    let lr = 0.16 * ref_norm / (e.rows as f32).sqrt();
    for step in 1..=steps + 1 {
        let mut nll = 0.0;
        g.fill(0.0);
        for example in &rewrite {
            nll += e.grad(example, layer, &delta, &mut temp)? / rewrite.len() as f32;
            for (g, t) in g.iter_mut().zip(&temp) {
                *g += t / rewrite.len() as f32;
            }
        }
        if step == 1 {
            initial_loss = nll;
        }
        if nll < best_loss {
            best.clone_from(&delta);
            best_loss = nll;
        }
        if step > steps || nll < 0.05 {
            break;
        }
        e.grad(&kl, layer, &delta, &mut temp)?;
        let penalty = 0.01 / (ref_norm * norm(&delta).max(1e-5));
        for i in 0..e.rows {
            let g = g[i] + 0.0625 * temp[i] + penalty * delta[i];
            m[i] = 0.9 * m[i] + 0.1 * g;
            v[i] = 0.999 * v[i] + 0.001 * g * g;
            let m_hat = m[i] / (1.0 - 0.9f32.powi(step as i32));
            let v_hat = v[i] / (1.0 - 0.999f32.powi(step as i32));
            delta[i] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
        }
        let scale = 1.0f32.min(4.0 * ref_norm / norm(&delta));
        delta.iter_mut().for_each(|x| *x *= scale);
    }
    Ok((
        ref_value.iter().zip(best).map(|(a, b)| a + b).collect(),
        [initial_loss, best_loss],
    ))
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn hex(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02x}")).collect()
}

fn read_cache(path: &Path, width: usize) -> Result<Vec<Vec<f64>>> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len() as usize;
    let stride = width * 4;
    ensure!(
        size != 0 && size % stride == 0 && size / stride <= 16 * MAX_TOKENS,
        "invalid moment cache length"
    );
    let mut bytes = vec![0; size];
    f.read_exact(&mut bytes)?;
    let weight = (size as f64 / stride as f64).sqrt().recip();
    let result: Vec<Vec<f64>> = bytes
        .chunks_exact(stride)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v) as f64 * weight)
                .collect()
        })
        .collect();
    ensure!(
        result.iter().flatten().all(|x| x.is_finite()),
        "invalid moment cache value"
    );
    Ok(result)
}

fn moment(e: &Engine, root: &Path, hash: &str, layer: usize) -> Result<Vec<Vec<f64>>> {
    fs::create_dir_all(root)?;
    let backend = if e.device.is_cuda() { "cuda-" } else { "" };
    let file = root.join(format!("v1-{backend}{hash}-{SAMPLE_SHA}-{layer}.f32"));
    if file.exists() {
        return read_cache(&file, e.cols);
    }
    let staged = file.with_extension(format!("{}.new", std::process::id()));
    let guard = RemoveFile(staged);
    let mut writer = std::io::BufWriter::new(File::create(&guard.0)?);
    for line in SAMPLE.lines() {
        let tokens = e.tokenize(line, true)?;
        for tokens in tokens.chunks(e.n_ctx) {
            let readout = e.sample(tokens, layer, layer)?;
            for value in readout.keys.iter().flatten() {
                ensure!(value.is_finite(), "second moment contains a non-finite key");
                writer.write_all(&value.to_le_bytes())?;
            }
        }
    }
    writer.flush()?;
    fs::rename(&guard.0, &file)?;
    // This is a recomputable, worker-private cache. Bound its persistent size.
    let mut entries = fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("v1-")
                && name.ends_with(".f32")
                && e.file_type().is_ok_and(|ty| ty.is_file())
                && e.path() != file
        })
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect::<Vec<_>>();
    entries.sort_unstable_by(|a, b| b.cmp(a));
    for (_, old) in entries.into_iter().skip(MAX_CACHED_MOMENTS - 1) {
        let _ = fs::remove_file(old);
    }
    read_cache(&file, e.cols)
}

struct RemoveFile(PathBuf);
impl Drop for RemoveFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

type Located = Vec<Vec<(Vec<i32>, usize)>>;

fn solve(
    e: &Engine,
    req: &Request,
    loc: &Located,
    targets: &[Vec<f32>],
    root: &Path,
    hash: &str,
    layer: usize,
    z_layer: usize,
    left: usize,
) -> Result<Update> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mm = moment(e, root, hash, layer)?;
    let ridge = (mm.iter().flatten().map(|x| x * x).sum::<f64>() / e.cols as f64).max(1e-10);
    let mut lookup = Vec::new();
    for (contexts, target) in loc.iter().zip(targets) {
        let mut residual = Vec::new();
        let mut context_keys = Vec::new();
        for (i, (tok, pos)) in contexts.iter().enumerate() {
            let c = e.sample(tok, layer, z_layer)?;
            let k = &c.keys[*pos];
            if i == 0 {
                let v = if req.method == Method::Rome {
                    &c.mlp[*pos]
                } else {
                    &c.residual[*pos]
                };
                residual = target
                    .iter()
                    .zip(v)
                    .map(|(a, b)| (a - b) / left as f32)
                    .collect();
                if req.method == Method::Rome {
                    lookup = to_f64(k);
                }
            }
            context_keys.push(k.clone());
        }
        keys.push(edit_solve::mean_key(&context_keys, &CONTEXT_GROUPS));
        values.push(residual);
    }
    let keys = match req.method {
        Method::Rome => edit_solve::rome(keys.remove(0), &lookup, &mm, ridge * 0.01),
        Method::Memit => edit_solve::memit(&keys, &mm, 32.0, ridge),
        Method::Alphaedit => edit_solve::alphaedit(&keys, &mm, ridge),
    };
    let keys: Vec<Vec<f32>> = keys
        .iter()
        .map(|v| v.iter().map(|x| *x as f32).collect())
        .collect();
    ensure!(
        keys.iter().chain(&values).flatten().all(|x| x.is_finite()),
        "ill-conditioned edit solve"
    );
    Ok(Update {
        layer,
        keys,
        values,
    })
}

pub(crate) fn run(args: Args) -> Result<()> {
    trace::check_threads(args.threads)?;
    let device = args.backend.select()?;
    ensure!(!args.out.exists(), "output already exists");
    let file = File::open(&args.request)?;
    ensure!(file.metadata()?.len() <= 65536, "edit request too large");
    let req: Request = serde_json::from_reader(BufReader::new(file))?;
    ensure!(
        (1..=32).contains(&req.steps)
            && (1..=8).contains(&req.facts.len())
            && (1..=3).contains(&req.layers.len())
            && req.layers.windows(2).all(|a| a[0] < a[1]),
        "invalid edit layer, fact, or step count"
    );
    match req.method {
        Method::Rome => ensure!(
            req.layers.len() == 1 && req.facts.len() == 1,
            "ROME needs one layer and fact"
        ),
        Method::Memit => ensure!(req.layers.len() >= 2, "MEMIT needs two or three layers"),
        Method::Alphaedit => {}
    }
    for f in &req.facts {
        ensure!(
            !f.subject.trim().is_empty()
                && f.subject.len() <= 256
                && !f.target.trim().is_empty()
                && f.target.len() <= 256
                && f.prompt.len() <= 1024
                && f.prompt.matches("{}").count() == 1,
            "invalid edit fact"
        );
    }
    let spec = ModelFile::open(&args.model)?;
    ensure!(
        matches!(spec.s.arch.as_str(), "llama" | "qwen2") && spec.s.n_experts == 0,
        "edit requires a dense Llama or Qwen2 model"
    );
    let z_layer = *req.layers.last().unwrap();
    ensure!(z_layer < spec.s.n_layers, "edit layer outside model");
    let cols = spec
        .g
        .uint(&format!("{}.feed_forward_length", spec.s.arch))
        .context("missing FFN width")? as usize;
    ensure!((1..=32768).contains(&cols), "FFN width outside edit limit");
    let mut bf16 = None;
    for layer in &req.layers {
        let t = spec
            .g
            .tensor(&format!("blk.{layer}.ffn_down.weight"))
            .context("missing FFN down")?;
        let low = match t.ty {
            ffi::ggml_type::GGML_TYPE_F32 => false,
            ffi::ggml_type::GGML_TYPE_BF16 => true,
            _ => bail!("edit requires F32 or BF16 down weights"),
        };
        ensure!(
            bf16.is_none_or(|prior| prior == low),
            "mixed edited weight dtypes"
        );
        bf16 = Some(low);
    }
    let bf16 = bf16.unwrap();
    ensure!(
        hex(&Sha256::digest(SAMPLE.as_bytes())) == SAMPLE_SHA,
        "sample hash mismatch"
    );
    let mut model_file = BufReader::new(File::open(&args.model)?);
    let mut hasher = Sha256::new();
    use std::io::BufRead;
    loop {
        let bytes = model_file.fill_buf()?;
        if bytes.is_empty() {
            break;
        }
        hasher.update(bytes);
        let count = bytes.len();
        model_file.consume(count);
    }
    let hash = hex(&hasher.finalize());
    let expanded = RemoveFile(
        args.out
            .with_extension(format!("{}.f32.gguf", std::process::id())),
    );
    // The original BF16 file is only read. Drop the temporary F32 file as
    // soon as the private non-mmap model has loaded into memory.
    let path = if bf16 {
        let a = CString::new(args.model.as_os_str().as_encoded_bytes())?;
        let b = CString::new(expanded.0.as_os_str().as_encoded_bytes())?;
        let mut params = unsafe { ffi::llama_model_quantize_default_params() };
        params.nthread = args.threads;
        params.ftype = ffi::llama_ftype_LLAMA_FTYPE_ALL_F32;
        params.pure_ = true;
        params.max_buf_size = 128 * 1024 * 1024;
        ensure!(
            unsafe { ffi::llama_model_quantize(a.as_ptr(), b.as_ptr(), &params) } == 0,
            "cannot expand BF16 model to F32"
        );
        &expanded.0
    } else {
        &args.model
    };
    let model = device.load_model(path, false)?;
    drop(expanded);
    let vocab = unsafe { ffi::llama_model_get_vocab(model) };
    let e = Engine {
        ll: capture::Llama {
            model,
            ctx: ptr::null_mut(),
        },
        device,
        threads: args.threads,
        n_ctx: MAX_TOKENS.min(unsafe { ffi::llama_model_n_ctx_train(model) } as usize),
        vocab,
        nv: unsafe { ffi::llama_vocab_n_tokens(vocab) } as usize,
        cols,
        rows: spec.s.hidden,
        bf16,
    };
    let loc: Located = req
        .facts
        .iter()
        .map(|f| {
            PREFIXES
                .iter()
                .map(|prefix| e.locate(&f.prompt, &f.subject, prefix))
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    // All second moments describe the unedited parent, never a partly edited copy.
    for &layer in &req.layers {
        moment(&e, &args.cache, &hash, layer)?;
    }
    let (targets, losses): (Vec<_>, Vec<_>) = req
        .facts
        .iter()
        .map(|f| optimize(&e, f, req.method, z_layer, req.steps))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let mut updates = Vec::new();
    for (i, &layer) in req.layers.iter().enumerate() {
        let remaining = req.layers.len() - i;
        let update = solve(
            &e,
            &req,
            &loc,
            &targets,
            &args.cache,
            &hash,
            layer,
            z_layer,
            remaining,
        )?;
        if remaining > 1 {
            e.apply(&update)?;
        }
        updates.push(update);
    }
    let result = serde_json::json!({
        "schema_version": 1,
        "model_sha256": hash,
        "covariance_sha256": SAMPLE_SHA,
        "method": req.method,
        "source_dtype": if bf16 {"bf16"} else {"f32"},
        "gradient_dtype": "f32",
        "provenance": e.device,
        "losses": losses,
        "updates": updates,
    });
    let output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    serde_json::to_writer(output, &result)?;
    Ok(())
}
