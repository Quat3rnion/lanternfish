// Generate FFI bindings for llama.cpp / ggml / gguf and link the shared libraries.
//
// LLAMA_INCLUDE_DIR / LLAMA_LIB_DIR select a llama.cpp source build; otherwise
// the system install (/usr/include, default linker path) is used.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/ffi/wrapper.h");
    println!("cargo:rerun-if-env-changed=LLAMA_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");

    let mut b = bindgen::Builder::default()
        .header("src/ffi/wrapper.h")
        .allowlist_function("llama_.*|ggml_.*|gguf_.*")
        .allowlist_type("llama_.*|ggml_.*|gguf_.*")
        .allowlist_var("GGML_.*|LLAMA_.*")
        .rustified_enum("ggml_type|gguf_type|llama_flash_attn_type")
        .derive_default(true)
        .layout_tests(false);
    if let Ok(inc) = env::var("LLAMA_INCLUDE_DIR") {
        for d in inc.split(':') {
            b = b.clang_arg(format!("-I{d}"));
        }
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    b.generate().expect("bindgen failed: is llama.h installed? set LLAMA_INCLUDE_DIR")
        .write_to_file(out.join("bindings.rs")).unwrap();

    if let Ok(lib) = env::var("LLAMA_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    }
    for l in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }

    // rust-embed needs the folder to exist even before the UI is built.
    let dist = PathBuf::from("web/dist");
    if !dist.join("index.html").exists() {
        std::fs::create_dir_all(&dist).unwrap();
        std::fs::write(dist.join("index.html"),
            "<!doctype html><p>UI not built. Run <code>npm --prefix web ci && npm --prefix web run build</code>, then rebuild traverse.</p>").unwrap();
    }
    println!("cargo:rerun-if-changed=web/dist");
}
