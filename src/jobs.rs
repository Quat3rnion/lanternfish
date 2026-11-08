//! Trace pipeline (capture -> lens -> viz.json) and a one-at-a-time run queue
//! for the server. Runs are sequential because a capture uses all CPU threads
//! and maps the whole model.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::capture::{self, Options};
use crate::model::ModelFile;
use crate::store::Store;
use crate::viz::{self, BuildStats, Lens};

#[derive(Deserialize, Clone, Debug)]
pub struct RunRequest {
    pub model: PathBuf,
    pub name: String,
    pub prompt: String,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub raw: bool,
    #[serde(default)]
    pub think: bool,
}
fn default_n() -> usize {
    64
}

pub struct TraceParams {
    pub threads: i32,
    pub labels: usize,
}

/// Run a full trace and save it. `progress(phase, fraction)` is called often.
pub fn trace(
    store: &Store,
    req: &RunRequest,
    p: &TraceParams,
    progress: &(dyn Fn(&str, f32) + Sync),
) -> Result<BuildStats> {
    store.dir(&req.name)?; // validate before the expensive part
    let m = ModelFile::open(&req.model)?;
    let opt = Options {
        n_predict: req.n,
        threads: p.threads,
        raw: req.raw,
        think: req.think,
        ubatch: 512,
    };
    let cap = capture::run(
        &req.model,
        m.s.hidden,
        m.s.n_layers,
        m.s.top_k,
        &req.prompt,
        &opt,
        progress,
    )?;
    progress("loading LM head", 0.0);
    let lens = Lens::load(&m)?;
    let (viz, stats) = viz::build(
        &cap,
        &m,
        &lens,
        &req.name,
        p.labels,
        &store.label_cache(&req.model),
        progress,
    )?;
    progress("saving", 1.0);
    store.save(&req.name, &viz)?;
    Ok(stats)
}

#[derive(Serialize, Clone, Debug)]
pub struct Run {
    pub id: String,
    pub name: String,
    pub model: PathBuf,
    pub state: &'static str,
    pub phase: String,
    pub progress: f32,
    pub log_tail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct Queue {
    pub runs: Arc<Mutex<Vec<Run>>>,
    tx: mpsc::Sender<(String, RunRequest)>,
}

impl Queue {
    pub fn start(store: Store, params: TraceParams) -> Self {
        let runs: Arc<Mutex<Vec<Run>>> = Arc::default();
        let (tx, rx) = mpsc::channel::<(String, RunRequest)>();
        let r2 = runs.clone();
        std::thread::spawn(move || {
            for (id, req) in rx {
                let update = |f: &dyn Fn(&mut Run)| {
                    if let Some(r) = r2.lock().unwrap().iter_mut().find(|r| r.id == id) {
                        f(r);
                    }
                };
                update(&|r| r.state = "running");
                let log = Mutex::new(Vec::<String>::new());
                let progress = |phase: &str, frac: f32| {
                    let mut l = log.lock().unwrap();
                    if l.last().is_none_or(|x| !x.starts_with(phase)) {
                        l.push(format!("{phase} ({})", crate::util::now_iso()));
                        if l.len() > 12 {
                            l.remove(0);
                        }
                    }
                    let tail = l.join("\n");
                    update(&|r| {
                        r.phase = phase.to_string();
                        r.progress = frac;
                        r.log_tail = tail.clone();
                    });
                };
                match trace(&store, &req, &params, &progress) {
                    Ok(s) => update(&|r| {
                        r.state = "done";
                        r.progress = 1.0;
                        r.phase = format!(
                            "{} tokens, lens agreement {:.3}, {} expert labels",
                            s.tokens, s.lens_agreement, s.experts_labelled
                        );
                    }),
                    Err(e) => update(&|r| {
                        r.state = "failed";
                        r.error = Some(format!("{e:#}"));
                    }),
                }
            }
        });
        Self { runs, tx }
    }

    pub fn submit(&self, req: RunRequest) -> String {
        let id = format!(
            "{}-{}",
            crate::util::now_iso().replace([':', '-'], ""),
            req.name
        );
        self.runs.lock().unwrap().insert(
            0,
            Run {
                id: id.clone(),
                name: req.name.clone(),
                model: req.model.clone(),
                state: "queued",
                phase: "waiting".into(),
                progress: 0.0,
                log_tail: String::new(),
                error: None,
            },
        );
        let _ = self.tx.send((id.clone(), req));
        id
    }
}
