// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use std::path::PathBuf;

const KERNEL_FILES: [&str; 1] = ["kernels/ffi.cu"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    let paths = std::fs::read_dir("kernels/").unwrap();
    for path in paths {
        let path = path.unwrap().path();
        println!(
            "cargo:rerun-if-changed={}", path.display()
        );
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let kernels = KERNEL_FILES.iter().collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(out_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose");

    let out_file = out_dir.join("libselectivescan.a");
    println!("cargo:warning={}", out_file.display());
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=selectivescan");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
