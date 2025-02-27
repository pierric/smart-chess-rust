use clap::{Parser, ValueEnum};
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::{thread_rng, Rng};
use std::boxed::Box;
use std::cmp::Eq;
use std::path::Path;
use std::cell::{Ref};
use trace::Trace;

mod game;
mod mcts;
mod trace;
mod hexapawn;
mod chess;
mod knightmoves;
mod queenmoves;
mod underpromotions;

use hexapawn::{ChessTS, ChessEP};

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum Opponent {
    NN,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    white_device: String,

    #[arg(long, default_value = "<not-specified>")]
    black_device: String,

    #[arg(long, value_enum, default_value_t = Opponent::NN)]
    black_type: Opponent,

    #[arg(long, default_value = "<not-specified>")]
    black_checkpoint: String,

    #[arg(short, long, default_value_t = 60)]
    rollout: i32,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 0.0)]
    cpuct: f32,

    #[arg(short, long)]
    white_checkpoint: String,

    #[arg(short, long, default_value = "01.json")]
    output: String,
}

type MctsCursor = mcts::Cursor<hexapawn::Step>;

trait Player {
    fn bestmove(&self, cursor: &mut MctsCursor, state: &hexapawn::Board) -> Option<usize>;
}

struct NNPlayer<G: game::Game<hexapawn::Board>> {
    n_rollout: i32,
    cpuct: f32,
    temperature: f32,
    game: G,
}

impl NNPlayer<ChessTS> {
    fn load(device: &str, checkpoint: &Path, n_rollout: i32, cpuct: f32, temperature: f32) -> Self {
        let device = match device {
            "cpu" => tch::Device::Cpu,
            "cuda" => tch::Device::Cuda(0),
            "mps" => tch::Device::Mps,
            _ => todo!("Unsupported device name"),
        };

        let chess = ChessTS {
            model: tch::CModule::load_on_device(checkpoint, device).unwrap(),
            device: device,
        };

        NNPlayer {
            game: chess,
            n_rollout: n_rollout,
            cpuct: cpuct,
            temperature: temperature,
        }
    }
}

impl NNPlayer<ChessEP> {
    fn load(device: &str, checkpoint: &Path, n_rollout: i32, cpuct: f32, temperature: f32) -> Self {
        let device = match device {
            "cpu" => tch::Device::Cpu,
            "cuda" => tch::Device::Cuda(0),
            "mps" => tch::Device::Mps,
            _ => todo!("Unsupported device name"),
        };

        let chess = ChessEP {
            model: aotinductor::ModelPackage::new(checkpoint.to_string_lossy().as_ref()).unwrap(),
            device: device,
        };

        NNPlayer {
            game: chess,
            n_rollout: n_rollout,
            cpuct: cpuct,
            temperature: temperature,
        }
    }

}

impl<G: game::Game<hexapawn::Board>> Player for NNPlayer<G> {
    fn bestmove(&self, cursor: &mut MctsCursor, state: &hexapawn::Board) -> Option<usize> {
        mcts::mcts(
            &self.game,
            cursor.arc(),
            &state,
            self.n_rollout,
            Some(self.cpuct),
            true,
        );

        let num_act_vec: Vec<i32> = cursor
            .current()
            .children
            .iter()
            .map(|n: &mcts::ArcRefNode<_>| n.borrow().num_act)
            .collect();
        if num_act_vec.len() == 0 {
            return None;
        }

        let temperature = self.temperature; // if cursor.current().depth < 30 {1.0} else {self.temperature};

        let choice: usize = if temperature == 0.0 {
            let max = num_act_vec.iter().max().unwrap();
            let indices: Vec<usize> = num_act_vec
                .iter()
                .enumerate()
                .filter(|a| a.1 == max)
                .map(|a| a.0)
                .collect();
            let n: usize = thread_rng().gen_range(0..indices.len());
            indices[n]
        } else {
            let power = 1.0 / temperature;
            let weights =
                WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
            weights.sample(&mut thread_rng())
        };

        Some(choice)
    }
}

fn step(choice: usize, cursor: &mut MctsCursor, state: &mut hexapawn::Board, trace: &mut Trace<hexapawn::Move, hexapawn::Outcome>) {
    let q_value = cursor.current().q_value;
    let all_children: Vec<(hexapawn::Move, i32, f32)> = cursor
        .current()
        .children
        .iter()
        .map(|n: &mcts::ArcRefNode<_>| {
            let n: Ref<'_, _> = n.borrow();
            (n.step.0.unwrap(), n.num_act, n.q_value)
        })
        .collect();

    cursor.navigate_down(choice);
    cursor.current_mut().reset();

    let step = cursor.current().step;
    let mov = step.0.unwrap();
    let turn = !step.1;
    println!("{}: {}", turn, mov);
    trace.push(Some(mov), q_value, all_children);
    state.step(&mov);
}

fn play_loop<W, B>(
    white: &W,
    black: &B,
    cursor: &mut MctsCursor,
    state: &mut hexapawn::Board,
    trace: &mut Trace<hexapawn::Move, hexapawn::Outcome>,
) where
    W: Player + ?Sized,
    B: Player + ?Sized,
{
    let mut count = 0;

    loop {
        let next = white.bestmove(cursor, state);
        if next.is_none() {
            break;
        }
        step(next.unwrap(), cursor, state, trace);
        if state.outcome().is_some() {
            break;
        }

        let next = black.bestmove(cursor, state);
        if next.is_none() {
            break;
        }
        step(next.unwrap(), cursor, state, trace);
        if state.outcome().is_some() {
            break;
        }

        count += 1;
        if count >= 150 {
            break;
        }
    }
}

type SomeNNPlayer = dyn Player;

fn load_checkpoint<P: AsRef<Path>>(
    path: P,
    device: &str,
    n_rollout: i32,
    cpuct: f32,
    temperature: f32,
) -> Box<dyn Player> {
    let path = path.as_ref();
    match path.extension().and_then(std::ffi::OsStr::to_str) {
        Some("pt") => Box::new(NNPlayer::<ChessTS>::load(
            device,
            path,
            n_rollout,
            cpuct,
            temperature,
        )) as Box<SomeNNPlayer>,
        Some("pt2") => Box::new(NNPlayer::<ChessEP>::load(
            device,
            path,
            n_rollout,
            cpuct,
            temperature,
        )) as Box<SomeNNPlayer>,
        _ => panic!("--white-checkpoint should be a path to onnx or pt file."),
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    unsafe {
        backtrace_on_stack_overflow::enable();
        //match libloading::Library::new("libtorchtrt.so") {
        //    Err(e) => println!("torch_tensorrt not found: {}", e),
        //    Ok(_) => (),
        //}
    }
    let mut args = Args::parse();

    if args.black_checkpoint != "<not-specified>" {
        args.black_type = Opponent::NN;
        args.black_device = args.white_device.clone();
    }
    let args = args;

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

    let white_checkpoint = Path::new(&args.white_checkpoint);
    let white = load_checkpoint(
        white_checkpoint,
        &args.white_device,
        args.rollout,
        args.cpuct,
        args.temperature,
    );

    use std::borrow::Borrow;

    match args.black_type {
        Opponent::NN => {
            let black = load_checkpoint(
                args.black_checkpoint,
                &args.black_device,
                args.rollout,
                args.cpuct,
                args.temperature,
            );
            println!("Players loaded.");
            play_loop::<SomeNNPlayer, SomeNNPlayer>(
                white.borrow(),
                black.borrow(),
                &mut cursor,
                &mut state,
                &mut trace,
            );
        }
    }

    let outcome = state.outcome();
    println!("{:?}", outcome);
    outcome.map(|o| trace.set_outcome(o));
    trace.save(&args.output);
}
