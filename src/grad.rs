//! Read-only target-token loss and native backward for a selected weight.
use std::path::PathBuf;
use std::ptr;

use anyhow::{Context, Result, ensure};
use serde_json::json;

use crate::{backward_backend::Backend, capture, ffi, model::ModelFile, trace};

const MAX_TOKENS: usize = 256;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    layer: usize,
    /// Vocabulary ID of the next token after the prompt.
    #[arg(long)]
    target_token: i32,
    #[arg(long)]
    out: PathBuf,
    /// Backend for the model graph. CUDA backward does not fall back to CPU.
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,
    #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = super::default_threads())]
    threads: i32,
}

pub(crate) fn run(args: Args) -> Result<()> {
    trace::check_threads(args.threads)?;
    let device = args.backend.select()?;
    let model = ModelFile::open(&args.model)?;
    ensure!(
        matches!(model.s.arch.as_str(), "llama" | "qwen2") && model.s.n_experts == 0,
        "grad requires a dense Llama or Qwen2 model"
    );
    ensure!(
        args.layer < model.s.n_layers,
        "gradient layer outside model"
    );
    let name = format!("blk.{}.ffn_down.weight", args.layer);
    let weight = model
        .g
        .tensor(&name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        weight.ty == ffi::ggml_type::GGML_TYPE_F32,
        "grad requires F32 model weights"
    );
    let rows = model.s.hidden;
    let cols = model
        .g
        .uint(&format!("{}.feed_forward_length", model.s.arch))
        .context("missing feed_forward_length")? as usize;
    let n = rows
        .checked_mul(cols)
        .context("gradient tensor too large")?;
    ensure!(
        n.checked_mul(4) == Some(weight.size),
        "unexpected {name} shape"
    );

    let raw = device.load_model(&args.model, true)?;
    let mut ll = capture::Llama {
        model: raw,
        ctx: ptr::null_mut(),
    };
    let vocab = unsafe { ffi::llama_model_get_vocab(raw) };
    let tokens = capture::tokenize(vocab, &args.prompt)?;
    let max = MAX_TOKENS.min(unsafe { ffi::llama_model_n_ctx_train(raw) } as usize);
    ensure!(
        !tokens.is_empty() && tokens.len() <= max,
        "grad prompt must contain 1..={max} tokens"
    );
    let nv = unsafe { ffi::llama_vocab_n_tokens(vocab) };
    ensure!(
        (0..nv).contains(&args.target_token),
        "target token outside vocabulary"
    );
    let params = device.context_params(tokens.len(), args.threads);
    ll.ctx = unsafe { ffi::llama_init_from_model(raw, params) };
    ensure!(
        !ll.ctx.is_null(),
        "cannot create gradient context on {}",
        device.name()
    );

    let mut loss = 0.0;
    let mut grad = vec![0.0; n];
    let status = unsafe {
        ffi::llama_lanternfish_gradient(
            ll.ctx,
            tokens.as_ptr(),
            tokens.len() as i32,
            args.target_token,
            args.layer as i32,
            &mut loss,
            grad.as_mut_ptr(),
            n,
        )
    };
    ensure!(
        status == 0,
        "native gradient failed on {} (status {status}); use --verbose for the llama.cpp error",
        device.name()
    );
    ensure!(
        loss.is_finite() && grad.iter().all(|v| v.is_finite()),
        "non-finite loss or gradient"
    );
    let (tensor, data) = trace::Payload::f32("gradient.f32".into(), vec![rows, cols], &grad);
    let manifest = serde_json::to_vec(&json!({
        "schema_version": trace::SCHEMA_VERSION,
        "kind": "gradient",
        "model": model.s,
        "tokens": tokens,
        "position": tokens.len() - 1,
        "target_token": args.target_token,
        "objective": "next_token_cross_entropy",
        "loss": loss,
        "layer": args.layer,
        "weight": name,
        "tensor": tensor,
        "provenance": device,
    }))?;
    trace::write_export(&args.out, &manifest, &[(tensor, data)])
}
