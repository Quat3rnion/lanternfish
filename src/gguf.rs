//! GGUF metadata and tensor access through ggml's own gguf reader.
//! Dequantization uses ggml's `to_float` type traits, so every quant type that
//! the installed ggml supports works here with no extra code.
use std::ffi::{CStr, CString};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

use crate::ffi;

pub struct Gguf {
    ctx: *mut ffi::gguf_context,
    map: Mmap,
    data_offset: usize,
}

// The gguf context is read-only after init.
unsafe impl Send for Gguf {}
unsafe impl Sync for Gguf {}

pub struct TensorInfo {
    pub ty: ffi::ggml_type,
    pub offset: usize,
    pub size: usize,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let c = CString::new(path.to_string_lossy().as_bytes())?;
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        let ctx = unsafe { ffi::gguf_init_from_file(c.as_ptr(), params) };
        if ctx.is_null() {
            bail!("not a readable GGUF file: {}", path.display());
        }
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let data_offset = unsafe { ffi::gguf_get_data_offset(ctx) };
        Ok(Self {
            ctx,
            map,
            data_offset,
        })
    }

    fn key(&self, k: &str) -> Option<i64> {
        let c = CString::new(k).ok()?;
        let id = unsafe { ffi::gguf_find_key(self.ctx, c.as_ptr()) };
        (id >= 0).then_some(id)
    }

    pub fn str(&self, k: &str) -> Option<String> {
        let id = self.key(k)?;
        unsafe {
            if ffi::gguf_get_kv_type(self.ctx, id) != ffi::gguf_type::GGUF_TYPE_STRING {
                return None;
            }
            Some(
                CStr::from_ptr(ffi::gguf_get_val_str(self.ctx, id))
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    pub fn uint(&self, k: &str) -> Option<u64> {
        use ffi::gguf_type::*;
        let id = self.key(k)?;
        unsafe {
            Some(match ffi::gguf_get_kv_type(self.ctx, id) {
                GGUF_TYPE_UINT32 => ffi::gguf_get_val_u32(self.ctx, id) as u64,
                GGUF_TYPE_INT32 => ffi::gguf_get_val_i32(self.ctx, id) as u64,
                GGUF_TYPE_UINT64 => ffi::gguf_get_val_u64(self.ctx, id),
                _ => return None,
            })
        }
    }

    pub fn tensor(&self, name: &str) -> Option<TensorInfo> {
        let c = CString::new(name).ok()?;
        let id = unsafe { ffi::gguf_find_tensor(self.ctx, c.as_ptr()) };
        if id < 0 {
            return None;
        }
        unsafe {
            Some(TensorInfo {
                ty: ffi::gguf_get_tensor_type(self.ctx, id),
                offset: self.data_offset + ffi::gguf_get_tensor_offset(self.ctx, id),
                size: ffi::gguf_get_tensor_size(self.ctx, id),
            })
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensor(name).is_some()
    }

    /// Dequantize `rows` rows of `row_len` elements, starting at row `first`.
    pub fn rows(&self, name: &str, row_len: usize, first: usize, rows: usize) -> Result<Vec<f32>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| anyhow!("tensor {name} not in GGUF"))?;
        let row_bytes = unsafe { ffi::ggml_row_size(t.ty, row_len as i64) };
        let start = t.offset + first * row_bytes;
        let end = start + rows * row_bytes;
        if end > t.offset + t.size || end > self.map.len() {
            bail!("{name}: row range {first}+{rows} outside the tensor");
        }
        let src = &self.map[start..end];
        let mut out = vec![0f32; rows * row_len];
        to_float(t.ty, src, &mut out).with_context(|| format!("dequantize {name}"))?;
        Ok(out)
    }
}

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe { ffi::gguf_free(self.ctx) }
    }
}

pub fn to_float(ty: ffi::ggml_type, src: &[u8], out: &mut [f32]) -> Result<()> {
    use ffi::ggml_type::*;
    match ty {
        GGML_TYPE_F32 => {
            for (o, c) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        _ => {
            let traits = unsafe { &*ffi::ggml_get_type_traits(ty) };
            let f = traits
                .to_float
                .ok_or_else(|| anyhow!("ggml has no to_float for type {ty:?}"))?;
            // ggml dequantizes whole blocks; split into large chunks for rayon.
            let blck = unsafe { ffi::ggml_blck_size(ty) } as usize;
            let bsz = unsafe { ffi::ggml_type_size(ty) };
            let per = (1 << 16) / blck * blck;
            use rayon::prelude::*;
            out.par_chunks_mut(per).enumerate().for_each(|(i, o)| {
                let s = &src[i * per / blck * bsz..];
                unsafe { f(s.as_ptr() as *const _, o.as_mut_ptr(), o.len() as i64) }
            });
        }
    }
    Ok(())
}
