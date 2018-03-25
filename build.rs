
extern crate cc;

fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    std::env::set_var("CXX", "clang-3.8");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("cudakernel.cu")
        .compile("libcudakernel.a");
}
