
extern crate cc;

use std::process::Command;


fn get_ccbin() -> Result<String, &'static str> {
    // Try gcc
    println!("{:?}", Command::new("gcc").arg("--version").output().expect("g++ failed"));
    // Try clang
    let output = Command::new("ls").arg("/usr/bin/").output().expect("clang failed");
    println!("{:?}", output);
    let output = String::from_utf8(output.stdout).unwrap();
    let mut output = output.as_str();
    loop {
        let idx = match output.find("clang-") {
            Some(idx) => {
                let out: Result<i32, _> = str::parse(&output[idx+6..idx+7]);
                match out {
                    Ok(_) => {
                        output = &output[idx..].split("\n").next().unwrap();
                        break;
                    },
                    _ => {},
                }
                idx
            },
            None => { return Err("Couldn't find any installed cuda-compatible C compiler"); },
        };
        output = &output[idx+6..]
    }
    Ok(output.to_string())
}

fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    let ccbin = get_ccbin().unwrap();
    std::env::set_var("CXX", ccbin);

    cc::Build::new()
        .cuda(true)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("kernel/vectorkernel.cu")
        .compile("libvectorkernel.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("kernel/matrixkernel.cu")
        .compile("libmatrixkernel.a");
}
