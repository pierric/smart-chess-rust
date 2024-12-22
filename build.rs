fn main() {
    if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
}
