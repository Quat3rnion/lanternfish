//! Model-level facts read from GGUF metadata, plus the role table that maps
//! llama.cpp graph node names to what the viewer shows.
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde::Serialize;

use crate::gguf::Gguf;

/// Graph node base names per role, in priority order (first found wins).
pub const ROLES: &[(&str, &[&str])] = &[
    ("residual", &["l_out", "l_last"]),
    (
        "attn",
        &["linear_attn_out", "attn_out", "attn_output", "kqv_out"],
    ),
    ("ffn", &["ffn_out", "ffn_moe_out"]),
    ("experts", &["ffn_moe_topk"]),
    ("expert_w", &["ffn_moe_weights_norm", "ffn_moe_weights"]),
    ("shared_gate", &["shared_expert_gate_sigmoid"]),
];
/// Attention node names that mark a linear-attention (recurrent) layer.
pub const LINEAR_MARKERS: &[&str] = &["linear_attn_out"];

pub fn role_of(base: &str) -> Option<(&'static str, usize)> {
    ROLES
        .iter()
        .find_map(|(role, names)| names.iter().position(|n| *n == base).map(|p| (*role, p)))
}

#[derive(Serialize, Clone, Debug)]
pub struct Summary {
    pub arch: String,
    pub name: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub tied_head: bool,
    pub final_norm: bool,
    pub path: PathBuf,
}

pub struct ModelFile {
    pub g: Gguf,
    pub s: Summary,
}

impl ModelFile {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        let arch = g
            .str("general.architecture")
            .ok_or_else(|| anyhow!("no general.architecture"))?;
        let u = |k: &str| g.uint(&format!("{arch}.{k}")).unwrap_or(0) as usize;
        let s = Summary {
            name: g.str("general.name").unwrap_or_else(|| arch.clone()),
            n_layers: u("block_count"),
            hidden: u("embedding_length"),
            n_experts: u("expert_count"),
            top_k: u("expert_used_count"),
            tied_head: !g.has("output.weight"),
            final_norm: g.has("output_norm.weight"),
            path: path.to_path_buf(),
            arch,
        };
        if s.n_layers == 0 || s.hidden == 0 {
            return Err(anyhow!(
                "{}: missing block_count/embedding_length",
                path.display()
            ));
        }
        Ok(Self { g, s })
    }

    /// LM head as [vocab, hidden] f32 (tied embeddings fall back to token_embd).
    pub fn head(&self) -> Result<(Vec<f32>, usize)> {
        let name = if self.s.tied_head {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let t = self.g.tensor(name).ok_or_else(|| anyhow!("no {name}"))?;
        let row = unsafe { crate::ffi::ggml_row_size(t.ty, self.s.hidden as i64) };
        let vocab = t.size / row;
        Ok((self.g.rows(name, self.s.hidden, 0, vocab)?, vocab))
    }

    pub fn final_norm(&self) -> Result<Option<Vec<f32>>> {
        if !self.s.final_norm {
            return Ok(None);
        }
        Ok(Some(self.g.rows(
            "output_norm.weight",
            self.s.hidden,
            0,
            1,
        )?))
    }

    /// Expert `e` of layer `l` down projection as [hidden][inter] row-major.
    pub fn expert_down(&self, l: usize, e: usize) -> Result<(Vec<f32>, usize)> {
        let name = format!("blk.{l}.ffn_down_exps.weight");
        let t = self.g.tensor(&name).ok_or_else(|| anyhow!("no {name}"))?;
        let elems_per_expert = (t.size / self.s.n_experts)
            / unsafe { crate::ffi::ggml_type_size(t.ty) }
            * unsafe { crate::ffi::ggml_blck_size(t.ty) } as usize;
        let inter = elems_per_expert / self.s.hidden;
        Ok((
            self.g
                .rows(&name, inter, e * self.s.hidden, self.s.hidden)?,
            inter,
        ))
    }
}
