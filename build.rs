
extern crate cc;

fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    std::env::set_var("CXX", "clang-3.8");

    cc::Build::new()
        .cuda(true)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("vectorkernel.cu")
        .compile("libvectorkernel.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("matrixkernel.cu")
        .compile("libmatrixkernel.a");
}
