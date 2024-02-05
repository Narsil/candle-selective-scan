// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::path::PathBuf;
    let kernels = vec!["kernels/selective_scan/ffi.cu", "kernels/causal_conv1d/ffi.cu"];
    let watch: Vec<_> = std::fs::read_dir("kernels/")
        .unwrap()
        .map(|p| p.unwrap().path())
        .collect();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .watch(watch)
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
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=selectivescan");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    build_cuda()
}
