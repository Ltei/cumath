
extern crate cc;

use std::process::Command;
use std::str;


fn get_ccbin() -> Result<String, &'static str> {
    // Try gcc
    let output = Command::new("gcc").arg("--version").output();
    if let Ok(output) = output {
        if let Ok(output) = String::from_utf8(output.stdout) {
            if let Some(first_line) = output.split("\n").next() {
                let parsed: Vec<_> = first_line.split_whitespace().collect();
                let first_number = parsed[parsed.len()-2].split(".").next().unwrap().parse::<usize>();
                if let Ok(number) = first_number {
                    if number < 6 {
                        return Ok("g++".to_string())
                    }
                }
            }
        }
    }
    // Try clang
    let output = Command::new("ls").arg("/usr/bin/").output();
    if let Ok(output) = output {
        if let Ok(output) = String::from_utf8(output.stdout) {
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
                    None => { return Err("Couldn't find any installed cuda-compatible C compiler") },
                };
                output = &output[idx+6..]
            }
            return Ok(output.to_string())
        }
    }
    Err("Couldn't find any installed cuda-compatible C compiler")
}

fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    std::env::set_var("CXX", get_ccbin().unwrap());

    cc::Build::new()
        .cuda(true)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("kernel/vectorfragment.cu")
        .compile("libvectorfragment.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("kernel/vectorpacked.cu")
        .compile("libvectorpacked.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        .file("kernel/matrix.cu")
        .compile("libmatrix.a");
}
