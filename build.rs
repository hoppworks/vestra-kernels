fn main() {
    if std::env::var_os("CARGO_FEATURE_ONEDNN_EXPERIMENT").is_none() {
        return;
    }

    let root = std::env::var("ONEDNN_ROOT")
        .expect("onednn-experiment requires ONEDNN_ROOT to point at pinned oneDNN source");
    let lib_dir = std::env::var("ONEDNN_LIB_DIR")
        .expect("onednn-experiment requires ONEDNN_LIB_DIR containing libdnnl.so");
    let omp_lib_dir = std::env::var("OMP_LIB_DIR")
        .expect("onednn-experiment requires OMP_LIB_DIR containing the selected libomp.so");

    println!("cargo:rerun-if-changed=native/onednn_bridge.cpp");
    println!("cargo:rerun-if-env-changed=ONEDNN_ROOT");
    println!("cargo:rerun-if-env-changed=ONEDNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OMP_LIB_DIR");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-search=native={omp_lib_dir}");
    println!("cargo:rustc-link-lib=dylib=dnnl");
    println!("cargo:rustc-link-lib=dylib=omp");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{omp_lib_dir}");

    cc::Build::new()
        .cpp(true)
        .file("native/onednn_bridge.cpp")
        .include(format!("{root}/include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fopenmp=libomp")
        .compile("vestra_onednn_bridge");
}
