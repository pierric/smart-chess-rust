use std::path::PathBuf;

fn main() {
    if let Some(lib_path) = std::env::var_os("LIBTORCH") {
        let p = PathBuf::from(lib_path).join("lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", p.to_string_lossy());
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
}
