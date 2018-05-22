
extern crate cc;

use std::process::Command;
use std::str;
use std::io;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::convert::From;

fn main() {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");

    Command::new("nvcc").arg("--version").output().unwrap();

    std::env::set_var("CXX", get_ccbin().unwrap());

    cc::Build::new()
        .cuda(true)
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .file("kernels/vectorfragment.cu")
        .compile("libvectorfragment.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .file("kernels/vectorpacked.cu")
        .compile("libvectorpacked.a");
    cc::Build::new()
        .cuda(true)
        .cpp_link_stdlib(None)
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .file("kernels/matrix.cu")
        .compile("libmatrix.a");
}



#[derive(PartialEq, Debug)]
struct CumathBuildErrror {
    description: String,
}
impl CumathBuildErrror {
    fn new(description: String) -> CumathBuildErrror {
        CumathBuildErrror { description }
    }
}
impl Error for CumathBuildErrror {
    fn description(&self) -> &str {
        &self.description
    }
}
impl Display for CumathBuildErrror {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}
impl From<io::Error> for CumathBuildErrror {
    fn from(obj: io::Error) -> CumathBuildErrror {
        CumathBuildErrror { description: obj.description().to_owned() }
    }
}
impl From<std::string::FromUtf8Error> for CumathBuildErrror {
    fn from(obj: std::string::FromUtf8Error) -> CumathBuildErrror {
        CumathBuildErrror { description: obj.description().to_owned() }
    }
}
impl From<std::num::ParseIntError> for CumathBuildErrror {
    fn from(obj: std::num::ParseIntError) -> CumathBuildErrror {
        CumathBuildErrror { description: obj.description().to_owned() }
    }
}



fn get_ccbin() -> Result<String, CumathBuildErrror> {
    let gcc_result = try_with_gcc();
    if let Ok(gcc) = gcc_result {
        return Ok(gcc)
    }

    let clang_result = try_with_clang();
    if let Ok(clang) = clang_result {
        return Ok(clang)
    }

    Err(CumathBuildErrror::new(format!("Couldn't find any cuda-compatible C compiler\n- Couldn't use gcc because : {}\n- Couldn't use clang because : {}",
                                       gcc_result.unwrap_err().description(),
                                       clang_result.unwrap_err().description())))
}

fn get_default_gcc_version() -> Result<usize, CumathBuildErrror> {
    let output = Command::new("gcc").arg("--version").output()?;
    let output = String::from_utf8(output.stdout)?;
    match output.split("\n").next() {
        Some(output) => {
            let parsed: Vec<_> = output.split_whitespace().collect();
            let version = parsed[parsed.len()-2].split(".").next().unwrap().parse::<usize>()?;
            Ok(version)
        },
        None => {
            Err(CumathBuildErrror::new("Couldn't extract gcc version from gcc --version".to_owned()))
        }
    }
}
fn try_with_gcc() -> Result<String, CumathBuildErrror> {
    if get_default_gcc_version()? < 6 {
        Ok("g++".to_owned())
    } else {
        let output = Command::new("ls").arg("/usr/bin/").output()?;
        let output = String::from_utf8(output.stdout)?;
        println!("::::::::::::::::::::::::::::");
        for line in output.lines().rev() {
            if line.len() >= 5 && line[0..4].eq("gcc-") {
                if let Ok(version) = line[4..].parse::<f32>() {
                    if version < 6.0 {
                        return Ok(line.to_owned())
                    }
                }
            }
        }
        println!("::::::::::::::::::::::::::::");
        Err(CumathBuildErrror::new("g++ version >= 6".to_owned()))
    }
    /*let output = Command::new("gcc").arg("--version").output()?;
    let output = String::from_utf8(output.stdout)?;
    match output.split("\n").next() {
        Some(output) => {
            let parsed: Vec<_> = output.split_whitespace().collect();
            let version = parsed[parsed.len()-2].split(".").next().unwrap().parse::<usize>()?;
            if version < 6 {
                Ok("g++".to_owned())
            } else {
                Err(CumathBuildErrror::new("g++ version >= 6".to_owned()))
            }
        },
        None => {
            Err(CumathBuildErrror::new("Couldn't extract gcc version from gcc --version".to_owned()))
        }
    }*/
}
fn try_with_clang() -> Result<String, CumathBuildErrror> {
    let output = Command::new("ls").arg("/usr/bin/").output()?;
    let output = String::from_utf8(output.stdout)?;

    let mut output = output.as_str();
    loop {
        let idx = match output.find("clang-") {
            Some(idx) => {
                let out: Result<i32, _> = str::parse(&output[idx+6..idx+7]);
                if let Ok(_) = out {
                    output = &output[idx..].split("\n").next().unwrap();
                    break;
                }
                idx
            },
            None => { return Err(CumathBuildErrror::new("Couldn't find any installed clang version".to_owned())) },
        };
        output = &output[idx+6..]
    }
    return Ok(output.to_string())
}