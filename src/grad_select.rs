//! One selected derivative of final-position next-token cross entropy.
use std::ffi::CString;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::ptr;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{backward_backend::Backend, capture, ffi, model::ModelFile, trace};

pub(crate) fn tokenize(path: &Path, text: &str) -> Result<()> {
    ensure!(text.len() <= 65536, "oversize tokenizer input");
    capture::init_backends();
    let mut p = unsafe { ffi::llama_model_default_params() };
    p.vocab_only = true;
    p.n_gpu_layers = 0;
    let mut devices = [ptr::null_mut()];
    p.devices = devices.as_mut_ptr();
    let ll = capture::Llama {
        model: capture::load_model_with_params(path, p)?,
        ctx: ptr::null_mut(),
    };
    let tokens = capture::tokenize(unsafe { ffi::llama_model_get_vocab(ll.model) }, text)?;
    println!("{}", json!({"tokens":tokens}));
    Ok(())
}

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,
    #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = super::default_threads())]
    threads: i32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    tokens: Vec<i32>,
    target_id: i32,
    site: Site,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum Site {
    Weight {
        name: String,
    },
    Activation {
        layer: usize,
        role: Role,
        position: usize,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    LOut,
    AttnOut,
    FfnOut,
}

// Only constructed for a file in a newly created, exclusive output directory.
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn expand(source: &Path, dest: &Path, threads: i32) -> Result<()> {
    let a = CString::new(source.as_os_str().as_encoded_bytes())?;
    let b = CString::new(dest.as_os_str().as_encoded_bytes())?;
    let mut p = unsafe { ffi::llama_model_quantize_default_params() };
    p.nthread = threads;
    p.ftype = ffi::llama_ftype_LLAMA_FTYPE_ALL_F32;
    p.pure_ = true;
    p.max_buf_size = 128 << 20;
    ensure!(
        unsafe { ffi::llama_model_quantize(a.as_ptr(), b.as_ptr(), &p) } == 0,
        "cannot expand BF16 weights"
    );
    Ok(())
}

pub(crate) fn run(args: Args) -> Result<()> {
    trace::check_threads(args.threads)?;
    ensure!(
        fs::metadata(&args.spec)?.len() <= 65536,
        "oversize gradient spec"
    );
    let spec: Spec = serde_json::from_slice(&fs::read(&args.spec)?)?;
    ensure!(
        (1..=256).contains(&spec.tokens.len()),
        "expected 1..256 input tokens"
    );
    let model = ModelFile::open(&args.model)?;
    ensure!(
        matches!(model.s.arch.as_str(), "llama" | "qwen2") && model.s.n_experts == 0,
        "expected dense Llama or Qwen2"
    );
    let low = model.g.backward_needs_expansion()?;
    let (weight, layer, role, position, shape) = match &spec.site {
        Site::Weight { name } => (
            Some(CString::new(name.as_str())?),
            0,
            0,
            -1,
            model.g.shape(name)?,
        ),
        Site::Activation {
            layer,
            role,
            position,
        } => {
            ensure!(
                *layer < model.s.n_layers && *position < spec.tokens.len(),
                "activation site outside input"
            );
            (
                None,
                *layer as i32,
                match role {
                    Role::LOut => 0,
                    Role::AttnOut => 1,
                    Role::FfnOut => 2,
                },
                *position as i32,
                vec![model.s.hidden],
            )
        }
    };
    let n = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .context("gradient size overflow")?;
    ensure!(n > 0 && n.checked_mul(4).is_some(), "invalid gradient size");
    let device = args.backend.select()?;
    fs::create_dir(&args.out).context("expected a new output directory")?;
    let temp = Temporary(args.out.join("expanded.gguf"));
    let path = if low {
        expand(&args.model, &temp.0, args.threads)?;
        &temp.0
    } else {
        &args.model
    };
    let raw = device.load_model(path, true)?;
    let mut ll = capture::Llama {
        model: raw,
        ctx: ptr::null_mut(),
    };
    drop(temp);
    let nv = unsafe { ffi::llama_vocab_n_tokens(ffi::llama_model_get_vocab(raw)) };
    ensure!(
        spec.tokens
            .iter()
            .chain([&spec.target_id])
            .all(|i| (0..nv).contains(i)),
        "token ID outside vocabulary"
    );
    ensure!(
        spec.tokens.len() <= unsafe { ffi::llama_model_n_ctx_train(raw) } as usize,
        "input exceeds model context"
    );
    ll.ctx = unsafe {
        ffi::llama_init_from_model(raw, device.context_params(spec.tokens.len(), args.threads))
    };
    ensure!(!ll.ctx.is_null(), "gradient context unavailable");
    let mut loss = 0.;
    let mut gradient = vec![0.; n];
    let zeros = weight.is_none().then(|| vec![0.; n]);
    let status = unsafe {
        ffi::llama_lanternfish_grad_select(
            ll.ctx,
            spec.tokens.as_ptr(),
            spec.tokens.len() as i32,
            spec.target_id,
            weight.as_ref().map_or(ptr::null(), |w| w.as_ptr()),
            layer,
            role,
            position,
            zeros.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
            &mut loss,
            gradient.as_mut_ptr(),
            n,
        )
    };
    ensure!(
        status == 0,
        "gradient failed on {}; see --verbose",
        device.name()
    );
    ensure!(
        loss.is_finite() && gradient.iter().all(|x| x.is_finite()),
        "nonfinite result"
    );
    drop(ll);
    let mut hash = Sha256::new();
    let mut out = BufWriter::new(fs::File::create(args.out.join("gradient.f32"))?);
    for block in gradient.chunks(16384) {
        let bytes: Vec<_> = block.iter().flat_map(|x| x.to_le_bytes()).collect();
        hash.update(&bytes);
        out.write_all(&bytes)?;
    }
    out.flush()?;
    let sha256: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let meta = json!({"schema_version":1,"tokens":spec.tokens,"target_id":spec.target_id,"site":spec.site,
        "dtype":"f32_le","shape":shape,"loss":loss,"objective":"next_token_cross_entropy","sha256":sha256,
        "backend":args.backend,"placement":{"device":device.name(),"strict":true}});
    fs::write(
        args.out.join("gradient.json.part"),
        serde_json::to_vec(&meta)?,
    )?;
    fs::rename(
        args.out.join("gradient.json.part"),
        args.out.join("gradient.json"),
    )?;
    Ok(())
}
