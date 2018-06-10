
extern crate cc;
use std::env;


fn main() {

    if let Ok(cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    cc::Build::new().cuda(true)
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .file("kernels/vectorfragment.cu").compile("libvectorfragment.a");
    cc::Build::new().cuda(true).cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_52,code=sm_52")
        .flag("-gencode").flag("arch=compute_53,code=sm_53")
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .flag("-gencode").flag("arch=compute_60,code=sm_60")
        .flag("-gencode").flag("arch=compute_62,code=sm_62")
        .file("kernels/vectorpacked.cu").compile("libvectorpacked.a");
    cc::Build::new().cuda(true).cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_52,code=sm_52")
        .flag("-gencode").flag("arch=compute_53,code=sm_53")
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .flag("-gencode").flag("arch=compute_60,code=sm_60")
        .flag("-gencode").flag("arch=compute_62,code=sm_62")
        .file("kernels/matrix.cu").compile("libmatrix.a");
}