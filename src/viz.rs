//! Turn a capture into viz.json: logit lens per (layer, token), write energy,
//! expert routes and expert labels. The schema is the one the viewer reads
//! (see web/README.md and docs/FORMAT.md).
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::capture::{Capture, layer_kind};
use crate::model::ModelFile;

const LENS_CHUNK: usize = 32;

pub struct Lens {
    pub head: Vec<f32>,
    pub vocab: usize,
    pub hidden: usize,
    pub norm: Option<Vec<f32>>,
}

impl Lens {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let (head, vocab) = m.head()?;
        Ok(Self {
            head,
            vocab,
            hidden: m.s.hidden,
            norm: m.final_norm()?,
        })
    }

    /// logits = head @ (rmsnorm(h) * final_norm), for rows of h. Returns [rows, vocab].
    pub fn logits(&self, rows: &[&[f32]]) -> Vec<f32> {
        let (n, h, v) = (rows.len(), self.hidden, self.vocab);
        let mut x = vec![0f32; n * h];
        for (r, src) in rows.iter().enumerate() {
            let s = (src.iter().map(|a| a * a).sum::<f32>() / h as f32).sqrt() + 1e-6;
            for j in 0..h {
                x[r * h + j] = src[j] / s * self.norm.as_ref().map_or(1.0, |w| w[j]);
            }
        }
        let mut out = vec![0f32; n * v];
        // out[n, v] = x[n, h] @ head^T[h, v], split over vocabulary blocks.
        // Blocks write disjoint column ranges of `out`.
        const BLOCK: usize = 4096;
        let out_ptr = out.as_mut_ptr() as usize;
        let x_ptr = x.as_ptr() as usize;
        (0..v.div_ceil(BLOCK)).into_par_iter().for_each(|b| {
            let c0 = b * BLOCK;
            let w = BLOCK.min(v - c0);
            unsafe {
                matrixmultiply::sgemm(
                    n,
                    h,
                    w,
                    1.0,
                    x_ptr as *const f32,
                    h as isize,
                    1,
                    self.head.as_ptr().add(c0 * h),
                    1,
                    h as isize,
                    0.0,
                    (out_ptr as *mut f32).add(c0),
                    v as isize,
                    1,
                );
            }
        });
        out
    }

    pub fn project(&self, dir: &[f32]) -> Vec<f32> {
        self.head
            .par_chunks(self.hidden)
            .map(|row| row.iter().zip(dir).map(|(a, b)| a * b).sum())
            .collect()
    }
}

fn top_by(v: &[f32], k: usize, desc: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let cmp = |a: &usize, b: &usize| {
        if desc {
            v[*b].total_cmp(&v[*a])
        } else {
            v[*a].total_cmp(&v[*b])
        }
    };
    idx.select_nth_unstable_by(k, cmp);
    idx.truncate(k);
    idx.sort_by(cmp);
    idx
}

fn r4(x: f32) -> Value {
    json!((x as f64 * 10000.0).round() / 10000.0)
}

/// Dominant output direction of an expert's down projection: power iteration on W W^T.
fn expert_direction(w: &[f32], hidden: usize, inter: usize) -> (Vec<f32>, f32) {
    let mut v: Vec<f32> = (0..hidden)
        .map(|i| w[i * inter..(i + 1) * inter].iter().sum())
        .collect();
    let mut t = vec![0f32; inter];
    for _ in 0..6 {
        t.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..hidden {
            let vi = v[i];
            for (a, b) in t.iter_mut().zip(&w[i * inter..(i + 1) * inter]) {
                *a += b * vi;
            }
        }
        for i in 0..hidden {
            v[i] = w[i * inter..(i + 1) * inter]
                .iter()
                .zip(&t)
                .map(|(a, b)| a * b)
                .sum();
        }
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt() + 1e-12;
        v.iter_mut().for_each(|a| *a /= n);
    }
    let wt_v: f32 = (0..inter)
        .map(|j| {
            let s: f32 = (0..hidden).map(|i| w[i * inter + j] * v[i]).sum();
            s * s
        })
        .sum::<f32>()
        .sqrt();
    let wn = w.iter().map(|a| a * a).sum::<f32>().sqrt() + 1e-9;
    (v, wt_v / wn)
}

pub struct BuildStats {
    pub tokens: usize,
    pub lens_agreement: f32,
    pub experts_labelled: usize,
}

pub fn build(
    cap: &Capture,
    m: &ModelFile,
    lens: &Lens,
    name: &str,
    labels: usize,
    label_cache: &Path,
    progress: &(dyn Fn(&str, f32) + Sync),
) -> Result<(Value, BuildStats)> {
    let s = &m.s;
    let t_count = cap.tokens.len();
    let next: Vec<Option<i32>> = (0..t_count)
        .map(|i| cap.tokens.get(i + 1).map(|t| t.id))
        .collect();
    let mut agree = (0usize, 0usize);
    let mut mass: HashMap<(usize, i32), f32> = HashMap::new();
    let mut layers = Vec::with_capacity(s.n_layers);
    for (l, toks) in cap.layers.iter().enumerate() {
        progress("logit lens", l as f32 / s.n_layers as f32);
        let mut idx: Vec<usize> = toks
            .iter()
            .filter(|(_, st)| st.resid.is_some())
            .map(|(i, _)| *i)
            .collect();
        idx.sort();
        let mut per = BTreeMap::new();
        for chunk in idx.chunks(LENS_CHUNK) {
            let rows: Vec<&[f32]> = chunk
                .iter()
                .map(|i| toks[i].resid.as_deref().unwrap())
                .collect();
            let lg = lens.logits(&rows);
            let recs: Vec<(usize, Value, Option<bool>)> = chunk.par_iter().enumerate().map(|(j, &i)| {
                let row = &lg[j * lens.vocab..(j + 1) * lens.vocab];
                let st = &toks[&i];
                let top = top_by(row, 5, true);
                let mx = row[top[0]];
                let z: f64 = row.iter().map(|v| ((v - mx) as f64).exp()).sum();
                let lensv: Vec<Value> = top.iter()
                    .map(|&t| json!([cap.vocab.get(t).cloned().unwrap_or_default(), r4((((row[t] - mx) as f64).exp() / z) as f32)]))
                    .collect();
                let resid = st.resid.as_deref().unwrap();
                let mut rec = json!({
                    "lens": lensv,
                    "resid_rms": r4((resid.iter().map(|a| a * a).sum::<f32>() / resid.len() as f32).sqrt()),
                    "write": {"attn": st.attn.map(|a| r4(a.1)), "ffn": st.ffn.map(|a| r4(a.1))},
                });
                if st.streams_rms.len() > 1 {
                    rec["streams_rms"] = json!(st.streams_rms.iter().map(|x| r4(*x)).collect::<Vec<_>>());
                }
                let mut hit = None;
                if let Some(nx) = next[i] {
                    let target = row[nx as usize];
                    rec["next_rank"] = json!(row.iter().filter(|v| **v > target).count());
                    hit = Some(top[0] as i32 == nx);
                }
                if let Some(ex) = &st.experts {
                    rec["experts"] = json!(ex);
                    if let Some((_, w)) = &st.weights {
                        rec["weights"] = json!(w.iter().take(ex.len()).map(|x| r4(*x)).collect::<Vec<_>>());
                    }
                }
                if let Some(g) = st.shared_gate {
                    rec["shared_gate"] = r4(g);
                }
                (i, rec, hit)
            }).collect();
            for (i, rec, hit) in recs {
                if l + 1 == s.n_layers
                    && let Some(h) = hit
                {
                    agree.0 += h as usize;
                    agree.1 += 1;
                }
                per.insert(i.to_string(), rec);
            }
        }
        for st in toks.values() {
            if let (Some(ex), Some((_, w))) = (&st.experts, &st.weights) {
                for (e, x) in ex.iter().zip(w) {
                    *mass.entry((l, *e)).or_default() += x;
                }
            }
        }
        layers
            .push(json!({"layer": l, "kind": layer_kind(&cap.layer_attn_base[l]), "tokens": per}));
    }

    // expert labels for the experts that carry the most routing mass
    let mut cache: BTreeMap<String, Value> = std::fs::read(label_cache)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let mut heavy: Vec<((usize, i32), f32)> = mass.into_iter().collect();
    heavy.sort_by(|a, b| b.1.total_cmp(&a.1));
    heavy.truncate(labels);
    let mut experts = BTreeMap::new();
    if s.n_experts > 0 {
        let todo: Vec<(usize, i32)> = heavy
            .iter()
            .map(|x| x.0)
            .filter(|(l, e)| !cache.contains_key(&format!("{l}.{e}")))
            .collect();
        for (k, (l, e)) in todo.iter().enumerate() {
            progress("expert labels", k as f32 / todo.len().max(1) as f32);
            let Ok((w, inter)) = m.expert_down(*l, *e as usize) else {
                continue;
            };
            let (dir, share) = expert_direction(&w, s.hidden, inter);
            let lg = lens.project(&dir);
            let word = |i: &usize| cap.vocab.get(*i).cloned().unwrap_or_default();
            cache.insert(
                format!("{l}.{e}"),
                json!({
                    "pos": top_by(&lg, 6, true).iter().map(word).collect::<Vec<_>>(),
                    "neg": top_by(&lg, 6, false).iter().map(word).collect::<Vec<_>>(),
                    "rank1": r4(share),
                }),
            );
        }
        for ((l, e), _) in &heavy {
            let key = format!("{l}.{e}");
            if let Some(v) = cache.get(&key) {
                experts.insert(key, v.clone());
            }
        }
        std::fs::write(label_cache, serde_json::to_vec(&cache)?)?;
    }

    let agreement = if agree.1 > 0 {
        agree.0 as f32 / agree.1 as f32
    } else {
        0.0
    };
    let viz = json!({
        "name": name,
        "model": s,
        "n_layers": s.n_layers, "n_experts": s.n_experts, "top_k": s.top_k,
        "tokens": cap.tokens.iter().enumerate().map(|(i, t)| json!({"i": i, "piece": t.piece, "phase": t.phase, "id": t.id})).collect::<Vec<_>>(),
        "steps": cap.steps,
        "next_candidates": cap.next,
        "layers": layers,
        "experts": experts,
        "lens_note": "lm_head(final_norm(rmsnorm(sum of residual streams)))",
        "lens_last_layer_agreement": (agreement * 1000.0).round() / 1000.0,
        "captured": cap.captured,
        "created": crate::util::now_iso(),
        "prompt": cap.prompt,
        "tool": format!("lanternfish {}", env!("CARGO_PKG_VERSION")),
    });
    Ok((
        viz,
        BuildStats {
            tokens: t_count,
            lens_agreement: agreement,
            experts_labelled: experts.len(),
        },
    ))
}
