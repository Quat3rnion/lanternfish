//! Trace storage: <home>/traces/<name>/{viz.json, meta.json, prompt.txt}.
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::util::valid_name;

#[derive(Clone)]
pub struct Store {
    pub home: PathBuf,
}

impl Store {
    pub fn new(home: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(home.join("traces"))?;
        Ok(Self { home })
    }

    pub fn dir(&self, name: &str) -> Result<PathBuf> {
        if !valid_name(name) {
            bail!("invalid trace name {name:?}: use letters, digits, '_', '.', '-'");
        }
        Ok(self.home.join("traces").join(name))
    }

    pub fn label_cache(&self, model: &Path) -> PathBuf {
        let stem = model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        self.home.join(format!("expert-labels-{}.json", crate::util::slug(&stem)))
    }

    pub fn save(&self, name: &str, viz: &Value) -> Result<()> {
        let d = self.dir(name)?;
        std::fs::create_dir_all(&d)?;
        let meta = json!({
            "name": name,
            "model": {"name": viz["model"]["name"], "arch": viz["model"]["arch"],
                      "n_layers": viz["n_layers"], "n_experts": viz["n_experts"]},
            "tokens": viz["tokens"].as_array().map_or(0, |a| a.len()),
            "created": viz["created"],
            "lens_last_layer_agreement": viz["lens_last_layer_agreement"],
        });
        std::fs::write(d.join("prompt.txt"), viz["prompt"].as_str().unwrap_or(""))?;
        let tmp = d.join("viz.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(viz)?)?;
        std::fs::rename(tmp, d.join("viz.json"))?;
        std::fs::write(d.join("meta.json"), serde_json::to_vec(&meta)?)?;
        Ok(())
    }

    /// Newest first. Traces without meta.json (older tools) get one derived from viz.json.
    pub fn list(&self) -> Vec<Value> {
        let mut out: Vec<(std::time::SystemTime, Value)> = vec![];
        let Ok(rd) = std::fs::read_dir(self.home.join("traces")) else { return vec![] };
        for e in rd.flatten() {
            let d = e.path();
            let viz = d.join("viz.json");
            let Ok(md) = viz.metadata() else { continue };
            let meta = std::fs::read(d.join("meta.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .or_else(|| {
                    let v: Value = serde_json::from_slice(&std::fs::read(&viz).ok()?).ok()?;
                    let name = d.file_name()?.to_string_lossy().into_owned();
                    let m = json!({"name": name, "model": {"name": v["model"]["name"], "arch": v["model"]["arch"],
                        "n_layers": v["n_layers"], "n_experts": v["n_experts"]},
                        "tokens": v["tokens"].as_array().map_or(0, |a| a.len()), "created": v["created"]});
                    std::fs::write(d.join("meta.json"), serde_json::to_vec(&m).ok()?).ok();
                    Some(m)
                });
            if let Some(m) = meta {
                out.push((md.modified().unwrap_or(std::time::UNIX_EPOCH), m));
            }
        }
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out.into_iter().map(|x| x.1).collect()
    }

    pub fn viz_path(&self, name: &str) -> Option<PathBuf> {
        let p = self.dir(name).ok()?.join("viz.json");
        p.exists().then_some(p)
    }
}

/// GGUF files under the given directories (first shard only, no mmproj).
pub fn scan_models(dirs: &[PathBuf]) -> Vec<Value> {
    fn walk(d: &Path, depth: usize, out: &mut Vec<Value>) {
        if depth > 6 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                walk(&p, depth + 1, out);
            } else if ft.is_file() {
                let n = e.file_name().to_string_lossy().into_owned();
                let lower = n.to_lowercase();
                let shard_ok = !lower.contains("-of-0") || lower.contains("-00001-of-");
                if lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.starts_with("ggml-vocab") && shard_ok {
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    if size > 50 << 20 {
                        out.push(json!({"path": p, "name": n, "size_bytes": size}));
                    }
                }
            }
        }
    }
    let mut out = vec![];
    for d in dirs {
        walk(d, 0, &mut out);
    }
    out.sort_by_key(|v| v["size_bytes"].as_u64().unwrap_or(0));
    out
}
