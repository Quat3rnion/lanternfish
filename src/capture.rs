//! Run a model in llama.cpp (CPU) and collect per-token, per-layer state from
//! named graph nodes through the scheduler eval callback.
//!
//! Only reduced values are kept (summed residual, write RMS, expert ids and
//! weights), so memory grows as tokens x layers x hidden, not with every node.
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::ffi;
use crate::model::{LINEAR_MARKERS, role_of};

#[derive(Serialize, Clone, Debug)]
pub struct Token {
    pub id: i32,
    pub piece: String,
    pub phase: &'static str,
    pub pos: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct Step {
    pub first: usize,
    pub n: usize,
    pub phase: &'static str,
}

#[derive(Serialize, Clone, Debug)]
pub struct Cand {
    pub id: i32,
    pub piece: String,
    pub p: f32,
}

#[derive(Default, Clone, Debug)]
pub struct TokState {
    pub resid: Option<Vec<f32>>, // sum of residual streams
    pub streams_rms: Vec<f32>,
    pub attn: Option<(usize, f32)>, // (role priority, rms)
    pub ffn: Option<(usize, f32)>,
    pub experts: Option<Vec<i32>>,
    pub weights: Option<(usize, Vec<f32>)>,
    pub shared_gate: Option<f32>,
}

pub struct Capture {
    pub tokens: Vec<Token>,
    pub steps: Vec<Step>,
    pub next: Vec<Vec<Cand>>,
    pub layers: Vec<HashMap<usize, TokState>>,
    pub layer_attn_base: Vec<Option<(usize, String)>>,
    pub captured: Vec<String>,
    pub prompt: String,
    pub vocab: Vec<String>,
}

pub struct Options {
    pub n_predict: usize,
    pub threads: i32,
    pub raw: bool,
    pub think: bool,
    pub ubatch: usize,
}

/// Progress callback: (phase, fraction 0..1).
pub type Progress<'a> = &'a (dyn Fn(&str, f32) + Sync);

struct Collector {
    hidden: usize,
    top_k: usize,
    first: usize,
    n: usize,
    streams: Option<usize>,
    layers: Vec<HashMap<usize, TokState>>,
    layer_attn_base: Vec<Option<(usize, String)>>,
    seen: std::collections::BTreeSet<String>,
    probe: Option<Vec<(String, String, String, i64, i64)>>,
    buf: Vec<u8>,
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

unsafe extern "C" fn quiet_log(_l: ffi::ggml_log_level, _t: *const c_char, _u: *mut c_void) {}

pub fn silence_llama() {
    unsafe { ffi::llama_log_set(Some(quiet_log), std::ptr::null_mut()) };
}

unsafe extern "C" fn eval_cb(t: *mut ffi::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    let c = unsafe { &mut *(ud as *mut Collector) };
    let t = unsafe { &*t };
    let name = unsafe { CStr::from_ptr(t.name.as_ptr()) }.to_string_lossy();
    if let Some(p) = c.probe.as_mut() {
        if !ask && !c.seen.contains(name.as_ref()) {
            c.seen.insert(name.to_string());
            let n = unsafe { ffi::ggml_nelements(t) };
            let op = unsafe { CStr::from_ptr(ffi::ggml_op_name(t.op)) }.to_string_lossy().into_owned();
            let ty = unsafe { CStr::from_ptr(ffi::ggml_type_name(t.type_)) }.to_string_lossy().into_owned();
            p.push((name.to_string(), op, ty, t.ne[0], if t.ne[0] > 0 { n / t.ne[0] } else { 0 }));
        }
        return true;
    }
    let Some((base, layer)) = split_name(&name) else { return !ask };
    let Some((role, prio)) = role_of(base) else { return !ask };
    if ask {
        return true;
    }
    if layer >= c.layers.len() {
        return true;
    }
    let is_f32 = t.type_ == ffi::ggml_type::GGML_TYPE_F32;
    let is_i32 = t.type_ == ffi::ggml_type::GGML_TYPE_I32;
    if !(is_f32 || is_i32) || t.nb[0] != 4 {
        return true;
    }
    let nel = unsafe { ffi::ggml_nelements(t) } as usize;
    let nbytes = nel * 4;
    c.buf.resize(nbytes, 0);
    let tp = t as *const _ as *mut ffi::ggml_tensor;
    if unsafe { ffi::ggml_is_contiguous(t) } {
        unsafe { ffi::ggml_backend_tensor_get(tp, c.buf.as_mut_ptr() as *mut c_void, 0, nbytes) };
    } else {
        // Strided view (e.g. ffn_moe_topk = first k columns of an argsort): copy row by row.
        let row = t.ne[0] as usize * 4;
        let mut dst = 0;
        for i3 in 0..t.ne[3] as usize {
            for i2 in 0..t.ne[2] as usize {
                for i1 in 0..t.ne[1] as usize {
                    let off = i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
                    unsafe { ffi::ggml_backend_tensor_get(tp, c.buf.as_mut_ptr().add(dst) as *mut c_void, off, row) };
                    dst += row;
                }
            }
        }
    }
    c.seen.insert(base.to_string());
    let f: Vec<f32> = if is_f32 {
        c.buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    } else {
        Vec::new()
    };
    let iv: Vec<i32> = if is_i32 {
        c.buf.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    } else {
        Vec::new()
    };
    let mut ne0 = t.ne[0] as usize;
    if role == "expert_w" && ne0 == 1 && c.top_k > 0 {
        ne0 = c.top_k; // {1, k, T} layout
    }
    let mut rows = nel / ne0.max(1);
    let mut streams = 1;
    match role {
        "residual" | "attn" | "ffn" if ne0 != c.hidden => return true,
        _ => {}
    }
    if role == "residual" {
        if layer == 0 || c.streams.is_none() {
            // layer 0 is never pruned: its row count fixes the stream count for this step
            if rows % c.n == 0 {
                c.streams = Some(rows / c.n);
            }
        }
        streams = c.streams.unwrap_or(1);
        rows /= streams;
    }
    let toks: Vec<usize> = if rows == c.n {
        (c.first..c.first + c.n).collect()
    } else if rows == 1 {
        vec![c.first + c.n - 1] // llama.cpp prunes the last layer to the logit rows
    } else {
        return true;
    };
    let per_row = ne0 * streams;
    for (k, &ti) in toks.iter().enumerate() {
        let st = c.layers[layer].entry(ti).or_default();
        match role {
            "residual" => {
                let row = &f[k * per_row..(k + 1) * per_row];
                let mut sum = vec![0f32; ne0];
                st.streams_rms.clear();
                for s in 0..streams {
                    let x = &row[s * ne0..(s + 1) * ne0];
                    st.streams_rms.push(rms(x));
                    for (a, b) in sum.iter_mut().zip(x) {
                        *a += b;
                    }
                }
                st.resid = Some(sum);
            }
            "attn" | "ffn" => {
                let v = rms(&f[k * ne0..(k + 1) * ne0]);
                let slot = if role == "attn" { &mut st.attn } else { &mut st.ffn };
                if slot.is_none_or(|(p, _)| prio < p) {
                    *slot = Some((prio, v));
                }
                if role == "attn" {
                    let cur = &mut c.layer_attn_base[layer];
                    if cur.as_ref().is_none_or(|(p, _)| prio < *p) {
                        *cur = Some((prio, base.to_string()));
                    }
                }
            }
            "experts" => st.experts = Some(iv[k * ne0..(k + 1) * ne0].to_vec()),
            "expert_w" => {
                if st.weights.as_ref().is_none_or(|(p, _)| prio < *p) {
                    st.weights = Some((prio, f[k * ne0..(k + 1) * ne0].to_vec()));
                }
            }
            "shared_gate" => st.shared_gate = Some(f[k * ne0]),
            _ => {}
        }
    }
    true
}

pub fn split_name(name: &str) -> Option<(&str, usize)> {
    let (base, layer) = name.rsplit_once('-')?;
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None; // derived views such as "l_out-3 (reshaped)"
    }
    Some((base, layer.parse().ok()?))
}

struct Llama {
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
}

impl Drop for Llama {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                ffi::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                ffi::llama_model_free(self.model);
            }
        }
    }
}

/// Load ggml backends. Builds with GGML_BACKEND_DL keep the CPU backend in a
/// plugin; LANTERNFISH_BACKEND_DIR points at it (set in the Docker image).
fn init_backends() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        match std::env::var("LANTERNFISH_BACKEND_DIR") {
            Ok(d) => {
                let c = CString::new(d).unwrap();
                ffi::ggml_backend_load_all_from_path(c.as_ptr());
            }
            Err(_) => ffi::ggml_backend_load_all(),
        }
        ffi::llama_backend_init();
    });
}

fn load(path: &Path) -> Result<*mut ffi::llama_model> {
    init_backends();
    let mut mp = unsafe { ffi::llama_model_default_params() };
    mp.n_gpu_layers = 0;
    let c = CString::new(path.to_string_lossy().as_bytes())?;
    let m = unsafe { ffi::llama_model_load_from_file(c.as_ptr(), mp) };
    if m.is_null() {
        bail!("llama.cpp could not load {} (unsupported architecture?)", path.display());
    }
    Ok(m)
}

fn context(model: *mut ffi::llama_model, n_ctx: usize, opt: &Options, col: &mut Collector) -> Result<*mut ffi::llama_context> {
    let mut cp = unsafe { ffi::llama_context_default_params() };
    cp.n_ctx = n_ctx as u32;
    cp.n_batch = opt.ubatch as u32;
    cp.n_ubatch = opt.ubatch as u32;
    cp.n_threads = opt.threads;
    cp.n_threads_batch = opt.threads;
    cp.cb_eval = Some(eval_cb);
    cp.cb_eval_user_data = col as *mut Collector as *mut c_void;
    cp.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
    let ctx = unsafe { ffi::llama_init_from_model(model, cp) };
    if ctx.is_null() {
        bail!("could not create a llama.cpp context");
    }
    Ok(ctx)
}

fn piece(vocab: *const ffi::llama_vocab, t: i32) -> String {
    let mut buf = [0 as c_char; 256];
    let n = unsafe { ffi::llama_token_to_piece(vocab, t, buf.as_mut_ptr(), buf.len() as i32, 0, true) };
    if n <= 0 {
        return String::new();
    }
    let b: Vec<u8> = buf[..n as usize].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&b).into_owned()
}

fn tokenize(vocab: *const ffi::llama_vocab, text: &str) -> Result<Vec<i32>> {
    let mut toks = vec![0i32; text.len() + 16];
    let n = unsafe {
        ffi::llama_tokenize(vocab, text.as_ptr() as *const c_char, text.len() as i32, toks.as_mut_ptr(), toks.len() as i32, true, true)
    };
    if n < 0 {
        bail!("tokenization failed");
    }
    toks.truncate(n as usize);
    Ok(toks)
}

/// Render the model's own chat template (built-in llama.cpp formats).
pub fn render_prompt(model: *mut ffi::llama_model, user: &str, think: bool) -> String {
    let tmpl = unsafe { ffi::llama_model_chat_template(model, std::ptr::null()) };
    if tmpl.is_null() {
        return user.to_string();
    }
    let tmpl_s = unsafe { CStr::from_ptr(tmpl) }.to_string_lossy().into_owned();
    let role = CString::new("user").unwrap();
    let content = CString::new(user.replace('\0', "")).unwrap();
    let msg = ffi::llama_chat_message { role: role.as_ptr(), content: content.as_ptr() };
    let mut buf = vec![0 as c_char; user.len() * 2 + 4096];
    let n = unsafe { ffi::llama_chat_apply_template(tmpl, &msg, 1, true, buf.as_mut_ptr(), buf.len() as i32) };
    if n < 0 || n as usize > buf.len() {
        return user.to_string();
    }
    let b: Vec<u8> = buf[..n as usize].iter().map(|&c| c as u8).collect();
    let mut s = String::from_utf8_lossy(&b).into_owned();
    // Thinking models: close an empty think block unless thinking is wanted.
    if !think && tmpl_s.contains("<think>") && !s.trim_end().ends_with("</think>") {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}

pub fn run(path: &Path, hidden: usize, n_layers: usize, top_k: usize, user: &str, opt: &Options, progress: Progress) -> Result<Capture> {
    progress("loading model", 0.0);
    let model = load(path)?;
    let mut ll = Llama { model, ctx: std::ptr::null_mut() };
    let vocab = unsafe { ffi::llama_model_get_vocab(model) };
    let prompt = if opt.raw { user.to_string() } else { render_prompt(model, user, opt.think) };
    let mut toks = tokenize(vocab, &prompt)?;
    let nt = toks.len();
    if nt == 0 {
        bail!("empty prompt");
    }
    let mut col = Box::new(Collector {
        hidden, top_k, first: 0, n: 0, streams: None,
        layers: vec![HashMap::new(); n_layers],
        layer_attn_base: vec![None; n_layers],
        seen: Default::default(), probe: None, buf: Vec::new(),
    });
    ll.ctx = context(model, nt + opt.n_predict + 16, opt, &mut col)?;
    let nv = unsafe { ffi::llama_vocab_n_tokens(vocab) } as usize;
    let mut out = Capture {
        tokens: vec![], steps: vec![], next: vec![], layers: vec![], layer_attn_base: vec![],
        captured: vec![], prompt: prompt.clone(), vocab: vec![],
    };
    let top10 = |ctx: *mut ffi::llama_context| -> Vec<Cand> {
        let lg = unsafe { std::slice::from_raw_parts(ffi::llama_get_logits_ith(ctx, -1), nv) };
        let mut idx: Vec<usize> = (0..nv).collect();
        idx.select_nth_unstable_by(10, |a, b| lg[*b].total_cmp(&lg[*a]));
        idx.truncate(10);
        idx.sort_by(|a, b| lg[*b].total_cmp(&lg[*a]));
        let mx = lg[idx[0]];
        let z: f64 = lg.iter().map(|v| ((v - mx) as f64).exp()).sum();
        idx.iter().map(|&i| Cand { id: i as i32, piece: piece(vocab, i as i32), p: (((lg[i] - mx) as f64).exp() / z) as f32 }).collect()
    };
    let mut pos = 0;
    let mut i = 0;
    while i < nt {
        let n = opt.ubatch.min(nt - i);
        for k in 0..n {
            out.tokens.push(Token { id: toks[i + k], piece: piece(vocab, toks[i + k]), phase: "prefill", pos: pos + k });
        }
        col.first = i;
        col.n = n;
        col.streams = None;
        let b = unsafe { ffi::llama_batch_get_one(toks.as_mut_ptr().add(i), n as i32) };
        if unsafe { ffi::llama_decode(ll.ctx, b) } != 0 {
            bail!("llama_decode failed in prefill");
        }
        out.steps.push(Step { first: i, n, phase: "prefill" });
        let c = top10(ll.ctx);
        let nxt = c[0].id;
        out.next.push(c);
        pos += n;
        i += n;
        if i >= nt {
            toks.push(nxt);
        }
        progress("prefill", i as f32 / (nt + opt.n_predict) as f32);
    }
    for g in 0..opt.n_predict {
        let t = *toks.last().unwrap();
        out.tokens.push(Token { id: t, piece: piece(vocab, t), phase: "decode", pos });
        if unsafe { ffi::llama_vocab_is_eog(vocab, t) } {
            break;
        }
        col.first = pos;
        col.n = 1;
        let last = toks.len() - 1;
        let b = unsafe { ffi::llama_batch_get_one(toks.as_mut_ptr().add(last), 1) };
        if unsafe { ffi::llama_decode(ll.ctx, b) } != 0 {
            bail!("llama_decode failed in decode");
        }
        out.steps.push(Step { first: pos, n: 1, phase: "decode" });
        let c = top10(ll.ctx);
        toks.push(c[0].id);
        out.next.push(c);
        pos += 1;
        progress("decode", (nt + g + 1) as f32 / (nt + opt.n_predict) as f32);
    }
    progress("reading vocabulary", 1.0);
    out.vocab = (0..nv as i32).map(|t| piece(vocab, t)).collect();
    drop(ll);
    let col = *col;
    out.layers = col.layers;
    out.layer_attn_base = col.layer_attn_base;
    out.captured = col.seen.into_iter().collect();
    Ok(out)
}

pub fn layer_kind(base: &Option<(usize, String)>) -> &'static str {
    match base {
        Some((_, b)) if LINEAR_MARKERS.contains(&b.as_str()) => "linear",
        Some(_) => "attention",
        None => "unknown",
    }
}

/// One small forward pass that records every node: (name, op, type, ne0, rows).
pub fn probe(path: &Path, threads: i32) -> Result<Vec<(String, String, String, i64, i64)>> {
    let model = load(path)?;
    let mut ll = Llama { model, ctx: std::ptr::null_mut() };
    let vocab = unsafe { ffi::llama_model_get_vocab(model) };
    let mut col = Box::new(Collector {
        hidden: 0, top_k: 0, first: 0, n: 0, streams: None, layers: vec![], layer_attn_base: vec![],
        seen: Default::default(), probe: Some(vec![]), buf: vec![],
    });
    let opt = Options { n_predict: 0, threads, raw: true, think: false, ubatch: 64 };
    ll.ctx = context(model, 64, &opt, &mut col)?;
    let mut toks = tokenize(vocab, "def add(a, b):")?;
    let b = unsafe { ffi::llama_batch_get_one(toks.as_mut_ptr(), toks.len() as i32) };
    if unsafe { ffi::llama_decode(ll.ctx, b) } != 0 {
        return Err(anyhow!("probe decode failed"));
    }
    drop(ll);
    Ok(col.probe.take().unwrap())
}
