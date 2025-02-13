use std::path::PathBuf;

fn main() {
    let libtorch_path = std::env::var_os("LIBTORCH").map(|lib_path| PathBuf::from(lib_path).join("lib"));

    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            libtorch_path.map(|p| {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", p.to_string_lossy());
            });
        },
        "macos" => {
            pyo3_build_config::add_extension_module_link_args();
            libtorch_path.map(|p| {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p.to_string_lossy());
            });
        }
        _ => {},
    }

    println!("cargo:rustc-link-arg=-ltorch");
}
