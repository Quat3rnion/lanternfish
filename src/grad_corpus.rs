use std::collections::HashSet;
use std::ffi::CString;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::ptr;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{backward_backend::Backend, capture, ffi, grad_select, model::ModelFile, trace};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
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
    weights: Vec<String>,
    corpus_sha256: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}

pub(crate) fn run(args: Args) -> Result<()> {
    trace::check_threads(args.threads)?;
    ensure!(fs::metadata(&args.spec)?.len() <= 1 << 20, "oversize spec");
    let spec: Spec = serde_json::from_slice(&fs::read(&args.spec)?)?;
    ensure!(
        (1..=4096).contains(&spec.weights.len()),
        "expected 1..4096 weights"
    );
    ensure!(
        spec.weights.iter().collect::<HashSet<_>>().len() == spec.weights.len(),
        "duplicate selected weight"
    );
    ensure!(
        fs::metadata(&args.corpus)?.len() <= 16 << 20,
        "oversize corpus"
    );
    let corpus = fs::read_to_string(&args.corpus)?;
    let digest = hex(&Sha256::digest(corpus.as_bytes()));
    ensure!(digest == spec.corpus_sha256, "corpus digest mismatch");
    let model = ModelFile::open(&args.model)?;
    ensure!(
        matches!(model.s.arch.as_str(), "llama" | "qwen2") && model.s.n_experts == 0,
        "expected dense Llama or Qwen2"
    );
    let low = model.g.backward_needs_expansion()?;
    let mut names = Vec::new();
    let mut shapes = Vec::new();
    for name in &spec.weights {
        names.push(CString::new(name.as_str())?);
        shapes.push(model.g.shape(name)?);
    }
    drop(model);
    let device = args.backend.select()?;
    fs::create_dir(&args.out).context("expected a new output directory")?;
    let tmp = grad_select::Temporary(args.out.join("expanded.gguf"));
    let raw = if low {
        grad_select::expand(&args.model, &tmp.0, args.threads)?;
        device.load_model(&tmp.0, true)?
    } else {
        device.load_model(&args.model, true)?
    };
    let mut ll = capture::Llama {
        model: raw,
        ctx: ptr::null_mut(),
    };
    drop(tmp);
    let vocab = unsafe { ffi::llama_model_get_vocab(raw) };
    let max_ctx = (unsafe { ffi::llama_model_n_ctx_train(raw) } as usize).min(256);
    let mut tokens = Vec::new();
    let mut count = 0;
    for line in corpus.lines() {
        ensure!(
            !line.is_empty() && line.len() <= 65536,
            "empty or oversize corpus line"
        );
        let seq = capture::tokenize_special(vocab, line, false)?;
        ensure!(
            (1..=max_ctx + 1).contains(&seq.len()),
            "line exceeds context or has no tokens"
        );
        count += seq.len() - 1;
        tokens.push(seq);
    }
    ensure!(count > 0, "no next-token observations");
    let mut sums = Vec::new();
    for shape in &shapes {
        let n = shape
            .iter()
            .try_fold(1usize, |x, &d| x.checked_mul(d))
            .context("invalid shape")?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(n)
            .context("cannot allocate gradient sum")?;
        buf.resize(n, 0f64);
        sums.push(buf);
    }
    let mut targets: Vec<_> = names
        .iter()
        .zip(&mut sums)
        .map(|(name, sum)| ffi::llama_lanternfish_weight_sum {
            name: name.as_ptr(),
            sum: sum.as_mut_ptr(),
            elements: sum.len(),
        })
        .collect();
    let mut loss = 0.;
    for seq in &tokens {
        let n = seq.len() - 1;
        if n == 0 {
            continue;
        }
        ll.ctx = unsafe { ffi::llama_init_from_model(raw, device.context_params(n, args.threads)) };
        ensure!(!ll.ctx.is_null(), "gradient context unavailable");
        let mut value = 0.;
        let rc = unsafe {
            ffi::llama_lanternfish_grad_sequence(
                ll.ctx,
                seq.as_ptr(),
                n as i32,
                targets.as_mut_ptr(),
                targets.len(),
                &mut value,
            )
        };
        ensure!(
            rc == 0 && value.is_finite(),
            "corpus backward failed on {}; see --verbose",
            device.name()
        );
        unsafe { ffi::llama_free(ll.ctx) };
        ll.ctx = ptr::null_mut();
        loss += f64::from(value) * n as f64;
    }
    drop(ll);
    let mut gradients = Vec::new();
    for (i, sum) in sums.into_iter().enumerate() {
        let file = format!("gradient-{i:04}.f32");
        let mut out = BufWriter::new(fs::File::create(args.out.join(&file))?);
        let mut hash = Sha256::new();
        let mut bytes = Vec::with_capacity(65536);
        for chunk in sum.chunks(16384) {
            bytes.clear();
            for &x in chunk {
                let v = (x / count as f64) as f32;
                ensure!(v.is_finite(), "nonfinite gradient");
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            hash.update(&bytes);
            out.write_all(&bytes)?;
        }
        out.flush()?;
        gradients.push(json!({"file":file,"name":spec.weights[i],"shape":shapes[i],
                              "sha256":hex(&hash.finalize())}));
    }
    let meta = json!({"schema_version":1,"tokens":tokens,"corpus_sha256":digest,
        "observation_count":count,"objective":"mean_next_token_nll","dtype":"f32_le",
        "loss":loss/count as f64,"backend":args.backend,"placement":{"device":device.name(),"strict":true},
        "gradients":gradients});
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
