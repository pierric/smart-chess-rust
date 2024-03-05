extern crate tch;
extern crate clap;

use clap::Parser;

#[derive(Parser, Debug)]
#[command()]
struct Args {
    #[arg(short, long)]
    path: String,

    #[arg(short, long)]
    device: String,
}

fn main() {
    unsafe {
        match libloading::Library::new("libtorchtrt.so") {
            Err(e) => println!("torch_tensorrt not found: {}", e),
            Ok(_) => (),
        }
    }    
    let args = Args::parse();
    let device = match args.device.as_str() {
        "cpu"  => tch::Device::Cpu,
        "cuda" => tch::Device::Cuda(0),
        _      => todo!("Unsupported device name"),
    };

    let m = tch::CModule::load_on_device(args.path, device).unwrap();
}