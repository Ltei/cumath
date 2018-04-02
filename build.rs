
extern crate cc;

fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    std::env::set_var("CXX", "clang-3.8");

    /*let mut build = cc::Build::new();
    build.cuda(true)
        .flag("-gencode").flag("arch=compute_61,code=sm_61");

    build.file("vectorkernel.cu")
        .compile("libvectorkernel.a");

    build.file("matrixkernel.cu")
        .compile("libmatrixkernel.a");*/

    cc::Build::new()
        .cuda(true)
        //.flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("vectorkernel.cu")
        .compile("libvectorkernel.a");
    cc::Build::new()
        .cuda(true)
        //.flag("-cudart=shared")
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("matrixkernel.cu")
        .compile("libmatrixkernel.a");
}
