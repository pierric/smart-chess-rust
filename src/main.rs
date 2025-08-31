use clap::Parser;
use ort::execution_providers::*;
use std::path::Path;
use std::{thread, time};

mod backends;
mod chess;
mod game;
mod knightmoves;
mod mcts;
mod queenmoves;
mod trace;
mod underpromotions;

#[allow(non_camel_case_types)]
mod jina {
    tonic::include_proto!("jina");
}
mod docarray {
    tonic::include_proto!("docarray");
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = String::from("cuda"))]
    device: String,

    #[arg(short, long, default_missing_value = None)]
    rollout_factor: Option<f32>,

    #[arg(long, default_missing_value = None)]
    rollout_num: Option<i32>,

    #[arg(short, long, default_value_t = 100)]
    num_steps: i32,

    #[arg(short, long, default_value_t = String::from("trace.json"))]
    trace_file: String,

    #[arg(short, long, default_value_t = String::from("__no_checkpoint__"))]
    checkpoint: String,

    #[arg(long, default_value_t = String::from("__no_endpoint__"))]
    endpoint: String,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    cpuct: f32,

    #[arg(long, default_value_t = 30)]
    temperature_switch: i32,
}

fn main() {
    unsafe {
        backtrace_on_stack_overflow::enable();
        // for nvidia only. skipping as using rocm
        //match libloading::Library::new("libtorchtrt.so") {
        //    Err(e) => println!("torch_tensorrt not found: {}", e),
        //    Ok(_) => (),
        //}
    }

    let args = Args::parse();

    assert!(args.rollout_factor.is_none() || args.rollout_num.is_none());

    let device = match args.device.as_str() {
        "cpu" => tch::Device::Cpu,
        "cuda" => tch::Device::Cuda(0),
        "mps" => tch::Device::Mps,
        _ => todo!("Unsupported device name"),
    };

    let chess: Box<dyn game::Game<chess::BoardState>> = if args.checkpoint != "__no_checkpoint__" {
        match Path::new(&args.checkpoint)
            .extension()
            .and_then(|s| s.to_str())
        {
            Some("pt") => {
                tch::jit::set_tensor_expr_fuser_enabled(true);
                Box::new(backends::torch::ChessTS {
                    model: tch::CModule::load_on_device(args.checkpoint, device).unwrap(),
                    device: device,
                })
            }
            Some("pt2") => Box::new(backends::torch::ChessEP {
                model: aotinductor::ModelPackage::new(&args.checkpoint).unwrap(),
                device: device,
            }),
            Some("onnx") => {
                let rocm = ROCmExecutionProvider::default();
                if !rocm.is_available().unwrap() {
                    eprintln!("Please compile ONNX Runtime with ROCm!");
                    std::process::exit(1);
                }
                ort::init()
                    .with_execution_providers([
                        CUDAExecutionProvider::default().build(),
                        ROCmExecutionProvider::default().build(),
                        MIGraphXExecutionProvider::default().build(),
                    ])
                    .commit()
                    .unwrap();
                let session = ort::session::Session::builder()
                    .unwrap()
                    .commit_from_file(args.checkpoint)
                    .unwrap();
                Box::new(backends::onnx::ChessOnnx {
                    session: std::cell::RefCell::new(session),
                })
            }
            _ => panic!("unsupported checkpoint type."),
        }
    } else {
        Box::new(backends::grpc::ChessService::new(&args.endpoint))
    };

    //use pyo3::prelude::*;
    //use pyo3::types::{IntoPyDict, PyString};
    //use c_str_macro::c_str;
    //    let chess = game::Chess {
    //        model: {
    //            Python::with_gil(|py| {
    //                let code = c_str!(r#"
    //mod = nn.ChessModule19().cuda()
    //mod.load_state_dict(torch.load(checkpoint), strict=True)
    //mod.eval()
    //mod = torch.compile(mod, mode="reduce-overhead", fullgraph=True)"#);
    //                let locals = [
    //                    ("torch", py.import("torch").unwrap().as_ref()),
    //                    ("nn", py.import("nn").unwrap().as_ref()),
    //                    ("checkpoint", PyString::new(py, &args.checkpoint).as_any()),
    //                ].into_py_dict(py).unwrap();
    //                py.run(code, None, Some(&locals)).unwrap();
    //                locals.get_item("mod").unwrap().unwrap().unbind()
    //            })
    //        },
    //        device: args.device,
    //    };

    let mut trace = trace::Trace::new();

    let mut state = chess::BoardState::new();

    let (mut cursor, _root) = mcts::Cursor::new(mcts::Node {
        step: chess::Step(None, chess::Color::White),
        depth: 0,
        q_value: 0.,
        num_act: 0,
        uct: 0.,
        parent: None,
        children: Vec::new(),
    });
    let mut outcome = None;

    for i in 0..args.num_steps {
        let temperature = if i < args.temperature_switch {
            1.0
        } else {
            args.temperature
        };

        let rollout = match (args.rollout_factor, args.rollout_num) {
            (Some(v), None) => i32::min(300, (state.legal_moves().len() as f32 * v) as i32),
            (None, Some(v)) => v,
            (None, None) => 300,
            (Some(_), Some(_)) => panic!("both --rollout_factor and --rollout_num are specified."),
        };
        println!(
            "Rollout {} Temp {} Cpuct {} Turn {}",
            rollout,
            temperature,
            args.cpuct,
            cursor.current().step.1
        );
        mcts::mcts(
            chess.as_ref(),
            cursor.arc(),
            &state,
            rollout,
            Some(args.cpuct),
            true,
        );

        let (q_value, num_act_children) = {
            let node = cursor.current();
            let q_value = node.q_value;
            let num_act_children: Vec<(chess::Move, i32, f32, f32)> = node
                .children
                .iter()
                .map(|n| {
                    let n = n.borrow();
                    (n.step.0.unwrap(), n.num_act, n.q_value, n.uct)
                })
                .collect();
            (q_value, num_act_children)
        };

        match mcts::step(&mut cursor, &mut state, temperature) {
            None => {
                outcome = state.outcome();
                break;
            }
            Some(step) => {
                trace.push(step.0, q_value, num_act_children);
                println!("Step {}\n{}", i, state);
            }
        }

        if i > 100 {
            outcome = state.outcome();
            if outcome.is_some() {
                break;
            }
        }

        // slow down slightly so that the GPU works not too hard
        let dur = time::Duration::from_millis(400);
        thread::sleep(dur);
    }

    if outcome.is_some() {
        trace.set_outcome(outcome.unwrap());
    }
    trace.save(&args.trace_file);
}
