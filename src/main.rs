//! lanternfish: trace and visualize how a llama.cpp model routes each token
//! through its layers and experts, in prefill and in decoding.
mod backward_backend;
mod capture;
mod edit;
mod edit_solve;
mod ffi;
mod gguf;
mod grad;
mod grad_select;
mod jobs;
mod model;
#[cfg(feature = "ui")]
mod server;
mod store;
mod trace;
mod util;
mod viz;

use std::collections::BTreeMap;
use std::path::PathBuf;
#[cfg(feature = "ui")]
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::jobs::{RunRequest, TraceParams};
use crate::model::{ModelFile, ROLES};
use crate::store::Store;

#[derive(Parser)]
#[command(name = "lanternfish", version, about)]
struct Cli {
    /// Where traces and caches are kept.
    #[arg(long, env = "LANTERNFISH_HOME", global = true)]
    home: Option<PathBuf>,
    /// Show llama.cpp log output.
    #[arg(long, short, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Export selected prompt activations and logits to a new directory.
    Trace(trace::Args),
    /// Export a next-token gradient for a selected FFN down weight.
    Grad(grad::Args),
    /// Export the next-token loss gradient at a weight or activation site.
    GradSelect(grad_select::Args),
    /// Encode raw text with model special tokens. Do not apply a chat template.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Compute ROME, MEMIT or AlphaEdit low-rank factors without writing source weights.
    LocateEdit(edit::Args),
    /// Run one small forward pass and show which graph nodes fill each role.
    Probe {
        model: PathBuf,
        /// List every node base name, not only the roles.
        #[arg(long)]
        all: bool,
        #[arg(long, default_value_t = 8)]
        threads: i32,
    },
    /// Trace a prompt (prefill plus greedy decoding) and save viz.json.
    Run {
        model: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long)]
        prompt: String,
        /// Decode tokens after the prompt.
        #[arg(long, short, default_value_t = 64)]
        n: usize,
        /// Use the prompt as is, without the chat template.
        #[arg(long)]
        raw: bool,
        /// Leave thinking on for thinking models.
        #[arg(long)]
        think: bool,
        #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = default_threads())]
        threads: i32,
        /// Experts to label (most routing mass first).
        #[arg(long, default_value_t = 600)]
        labels: usize,
    },
    /// List saved traces.
    List,
    /// Serve the UI and the API.
    #[cfg(feature = "ui")]
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8791)]
        port: u16,
        /// Directories scanned for GGUF files that may be traced from the UI.
        #[arg(long = "models-dir", env = "LANTERNFISH_MODELS", value_delimiter = ':')]
        models_dir: Vec<PathBuf>,
        #[arg(long, env = "LANTERNFISH_THREADS", default_value_t = default_threads())]
        threads: i32,
        #[arg(long, default_value_t = 600)]
        labels: usize,
    },
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32 / 2)
        .unwrap_or(8)
        .max(1)
}

fn home(path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(|| {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share")
            });
        base.join("lanternfish")
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.verbose {
        capture::silence_llama();
    }
    let cmd = match cli.cmd {
        Cmd::Trace(args) => return trace::run(args),
        Cmd::Grad(args) => return grad::run(args),
        Cmd::GradSelect(args) => return grad_select::run(args),
        Cmd::Tokenize { model, text } => return grad_select::tokenize(&model, &text),
        Cmd::LocateEdit(args) => return edit::run(args),
        cmd => cmd,
    };
    let store = Store::new(home(cli.home))?;
    match cmd {
        Cmd::Trace(_)
        | Cmd::Grad(_)
        | Cmd::GradSelect(_)
        | Cmd::Tokenize { .. }
        | Cmd::LocateEdit(_) => unreachable!(),
        Cmd::Probe {
            model,
            all,
            threads,
        } => {
            let m = ModelFile::open(&model)?;
            println!("{}", serde_json::to_string(&m.s)?);
            let nodes = capture::probe(&model, threads)?;
            let mut bases: BTreeMap<String, usize> = BTreeMap::new();
            for (name, ..) in &nodes {
                if let Some((b, _)) = capture::split_name(name) {
                    *bases.entry(b.to_string()).or_default() += 1;
                }
            }
            for (role, names) in ROLES {
                let found: Vec<String> = names
                    .iter()
                    .filter_map(|n| bases.get(*n).map(|c| format!("{n} ({c} layers)")))
                    .collect();
                println!(
                    "  {role:12} {}",
                    if found.is_empty() {
                        "-- not found".into()
                    } else {
                        found.join(", ")
                    }
                );
            }
            if all {
                for (b, c) in &bases {
                    println!("    {b}: {c} layers");
                }
            }
            let tsv = store.home.join("probe.tsv");
            std::fs::write(
                &tsv,
                nodes
                    .iter()
                    .map(|(n, o, t, a, b)| format!("{n}\t{o}\t{t}\t{a}\t{b}\n"))
                    .collect::<String>(),
            )?;
            println!("full node list: {}", tsv.display());
        }
        Cmd::Run {
            model,
            name,
            prompt,
            n,
            raw,
            think,
            threads,
            labels,
        } => {
            let t0 = std::time::Instant::now();
            let last = std::sync::Mutex::new(String::new());
            let progress = |phase: &str, f: f32| {
                let mut l = last.lock().unwrap();
                if *l != phase {
                    eprintln!("[{:>6.1}s] {phase}", t0.elapsed().as_secs_f32());
                    *l = phase.to_string();
                }
                let _ = f;
            };
            let req = RunRequest {
                model,
                name: name.clone(),
                prompt,
                n,
                raw,
                think,
            };
            let s = jobs::trace(&store, &req, &TraceParams { threads, labels }, &progress)?;
            println!(
                "{}",
                serde_json::json!({"name": name, "tokens": s.tokens, "lens_agreement": s.lens_agreement,
                "experts_labelled": s.experts_labelled, "seconds": t0.elapsed().as_secs(),
                "path": store.dir(&name)?.join("viz.json")})
            );
        }
        Cmd::List => {
            for m in store.list() {
                println!(
                    "{}\t{}\t{} tokens\t{}",
                    m["name"].as_str().unwrap_or(""),
                    m["model"]["name"].as_str().unwrap_or(""),
                    m["tokens"],
                    m["created"].as_str().unwrap_or("")
                );
            }
        }
        #[cfg(feature = "ui")]
        Cmd::Serve {
            host,
            port,
            models_dir,
            threads,
            labels,
        } => {
            let queue = jobs::Queue::start(store.clone(), TraceParams { threads, labels });
            let app = server::App {
                store,
                queue,
                model_dirs: Arc::new(models_dir),
            };
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                eprintln!("lanternfish UI on http://{addr}/");
                axum::serve(listener, server::router(app))
                    .with_graceful_shutdown(async {
                        let _ = tokio::signal::ctrl_c().await;
                    })
                    .await?;
                anyhow::Ok(())
            })?;
        }
    }
    Ok(())
}
