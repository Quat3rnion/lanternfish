//! Device selection shared by the native gradient and factor commands.
use std::ffi::{CStr, OsStr};
use std::path::Path;
use std::process::Command;
use std::ptr;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{capture, ffi};

#[derive(Clone, Copy, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Backend {
    Cpu,
    Cuda,
}

#[derive(Serialize)]
pub(crate) struct Device {
    backend: Backend,
    device: String,
    #[serde(skip)]
    raw: ffi::ggml_backend_dev_t,
}

impl Backend {
    pub(crate) fn select(self) -> Result<Device> {
        if matches!(self, Self::Cuda) {
            ensure_cuda_precision()?;
        }
        capture::init_backends();
        let raw = unsafe {
            match self {
                Self::Cpu => ffi::ggml_backend_dev_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                ),
                Self::Cuda => {
                    let reg = ffi::ggml_backend_reg_by_name(c"CUDA".as_ptr());
                    ensure!(
                        !reg.is_null()
                            && CStr::from_ptr(ffi::ggml_backend_reg_name(reg)) == c"CUDA",
                        "cuda backend unavailable: no CUDA registry"
                    );
                    (0..ffi::ggml_backend_reg_dev_count(reg))
                        .map(|index| ffi::ggml_backend_reg_dev_get(reg, index))
                        .find(|&dev| {
                            matches!(
                                ffi::ggml_backend_dev_type(dev),
                                ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                                    | ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
                            )
                        })
                        .context("cuda backend unavailable: no visible CUDA device")?
                }
            }
        };
        ensure!(!raw.is_null(), "cpu backend unavailable");
        let device = unsafe { CStr::from_ptr(ffi::ggml_backend_dev_name(raw)) }
            .to_string_lossy()
            .into_owned();
        Ok(Device {
            backend: self,
            device,
            raw,
        })
    }
}

// cuBLAS reads these environment variables during initialization. Relaunch
// before backend discovery instead of mutating the running process environment.
fn ensure_cuda_precision() -> Result<()> {
    const ENV: [(&str, &str); 2] = [
        ("NVIDIA_TF32_OVERRIDE", "0"),
        ("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32"),
    ];
    if ENV
        .iter()
        .all(|(key, value)| std::env::var_os(key).as_deref() == Some(OsStr::new(value)))
    {
        return Ok(());
    }

    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args(std::env::args_os().skip(1)).envs(ENV);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(cmd.exec()).context("cannot relaunch with CUDA F32 precision")
    }
    #[cfg(not(unix))]
    {
        let status = cmd
            .status()
            .context("cannot relaunch with CUDA F32 precision")?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

impl Device {
    pub(crate) fn is_cuda(&self) -> bool {
        matches!(self.backend, Backend::Cuda)
    }

    pub(crate) fn name(&self) -> &str {
        &self.device
    }

    pub(crate) fn load_model(&self, path: &Path, mmap: bool) -> Result<*mut ffi::llama_model> {
        let mut params = unsafe { ffi::llama_model_default_params() };
        let mut devices = [
            if self.is_cuda() {
                self.raw
            } else {
                ptr::null_mut()
            },
            ptr::null_mut(),
        ];
        params.devices = devices.as_mut_ptr();
        params.use_extra_bufts = false;
        params.n_gpu_layers = 0;
        if self.is_cuda() {
            params.n_gpu_layers = i32::MAX;
            params.split_mode = ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE;
        }
        if !mmap {
            params.load_mode = ffi::llama_load_mode_LLAMA_LOAD_MODE_NONE;
        }
        capture::load_model_with_params(path, params)
            .with_context(|| format!("cannot load backward model on {}", self.name()))
    }

    pub(crate) fn context_params(&self, n: usize, threads: i32) -> ffi::llama_context_params {
        let mut params = capture::cpu_context_params(n, n, threads);
        params.op_offload = self.is_cuda();
        params.offload_kqv = self.is_cuda();
        params
    }
}
