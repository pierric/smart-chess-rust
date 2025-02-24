use clap::Parser;
use std::path::Path;

mod hexapawn;

mod chess;
mod knightmoves;
mod queenmoves;
mod underpromotions;

mod game;
mod mcts;
mod trace;

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

    #[arg(short, long)]
    checkpoint: String,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    cpuct: f32,
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

    let chess: Box<dyn game::Game<hexapawn::Board>> = match Path::new(&args.checkpoint).extension().and_then(|s| s.to_str()) {
        Some("pt") => Box::new(hexapawn::ChessTS {
            model: tch::CModule::load_on_device(args.checkpoint, device).unwrap(),
            device: device,
        }),
        Some("pt2") => Box::new(hexapawn::ChessEP {
            model: aotinductor::ModelPackage::new(&args.checkpoint).unwrap(),
            device: device,
        }),
        _ => panic!("unsupported checkpoint type.")
    };

    let mut trace = trace::Trace::new();

    let mut state = hexapawn::Board::new();

    let (mut cursor, _root) = mcts::Cursor::new(mcts::Node {
        step: hexapawn::Step(None, hexapawn::Color::White),
        depth: 0,
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    });
    let mut outcome = None;

    for i in 0..args.num_steps {
        let temperature = if i < 30 { 1.0 } else { args.temperature };

        let rollout = match (args.rollout_factor, args.rollout_num) {
            (Some(v), None) => i32::max(200, (state.legal_moves().len() as f32 * v) as i32),
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
        mcts::mcts(chess.as_ref(), cursor.arc(), &state, rollout, Some(args.cpuct), true);

        let (q_value, num_act_children) = {
            let node = cursor.current();
            let q_value = node.q_value;
            let num_act_children: Vec<(hexapawn::Move, i32, f32)> = node
                .children
                .iter()
                .map(|n| {
                    let n = n.borrow();
                    (n.step.0.unwrap(), n.num_act, n.q_value)
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
    }

    if outcome.is_some() {
        trace.set_outcome(outcome.unwrap());
    }
    trace.save(&args.trace_file);
}
