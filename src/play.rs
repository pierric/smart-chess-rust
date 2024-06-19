use clap::{Parser, ValueEnum};
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::{thread_rng, Rng};
use std::borrow::Borrow;
use std::boxed::Box;
use std::cmp::Eq;
use std::fs::create_dir;
use std::path::Path;
use std::ptr::NonNull;
use trace::Trace;
use uci::Engine;

mod chess;
mod game;
mod knightmoves;
mod mcts;
mod queenmoves;
mod trace;
mod underpromotions;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum Opponent {
    Stockfish,
    NN,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    white_device: String,

    #[arg(long, default_value = "<not-specified>")]
    black_device: String,

    #[arg(long, value_enum, default_value_t = Opponent::Stockfish)]
    black_type: Opponent,

    #[arg(long, default_value = "/usr/bin/stockfish")]
    stockfish_bin: String,

    #[arg(long, default_value_t = 0)]
    stockfish_level: i32,

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

type MctsCursor = mcts::CursorMut<(Option<chess::Move>, chess::Color)>;

trait Player {
    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize>;
}

struct StockfishPlayer {
    engine: Engine,
}

impl StockfishPlayer {
    fn load(stockfish_path: &str, skill_level: i32) -> Self {
        let engine = Engine::new(stockfish_path).unwrap();
        engine
            .set_option("Skill Level", &skill_level.to_string())
            .unwrap();
        StockfishPlayer { engine }
    }
}

impl Player for StockfishPlayer {
    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize> {
        self.engine.set_position(&state.fen()).unwrap();
        let next = self
            .engine
            .bestmove()
            .ok()
            .map(|m| chess::Move::from_uci(&m))?;

        let mut choice: Option<usize> = None;
        let legal_moves = game::State::legal_moves(state);
        let node = cursor.current();
        let parent = NonNull::new(node as *mut _);
        node.children.clear();
        for (index, mov) in legal_moves.into_iter().enumerate() {
            let mut num_act = 0;
            if mov.0 == Some(next) {
                choice = Some(index);
                num_act = 1;
            }
            node.children.push(Box::new(mcts::Node {
                step: mov,
                depth: node.depth + 1,
                q_value: 0.,
                num_act: num_act,
                parent: parent,
                children: Vec::new(),
            }));
        }
        assert!(choice.is_some());
        choice
    }
}

struct NNPlayer<G: game::Game<chess::BoardState>> {
    n_rollout: i32,
    cpuct: f32,
    temperature: f32,
    game: G,
}

impl NNPlayer<game::ChessTS> {
    fn load(device: &str, checkpoint: &Path, n_rollout: i32, cpuct: f32, temperature: f32) -> Self {
        let device = match device {
            "cpu" => tch::Device::Cpu,
            "cuda" => tch::Device::Cuda(0),
            _ => todo!("Unsupported device name"),
        };

        let chess = game::ChessTS {
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

impl NNPlayer<game::ChessOnnx> {
    fn load(device: &str, checkpoint: &Path, n_rollout: i32, cpuct: f32, temperature: f32) -> Self {
        let device: &[ort::ExecutionProviderDispatch] = &match device {
            "cpu" => [ort::CPUExecutionProvider::default().build()],
            "cuda" => [ort::CUDAExecutionProvider::default().build()],
            _ => todo!("Unsupported device name"),
        };

        let session = move || -> ort::Result<ort::Session> {
            ort::init()
                .with_name("smart-chess")
                .with_execution_providers(device)
                .commit()?;
            ort::Session::builder()?
                .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
                .with_intra_threads(1)?
                .with_model_from_file(checkpoint)
        }();

        let chess = game::ChessOnnx {
            session: session.unwrap(),
        };

        NNPlayer {
            game: chess,
            n_rollout: n_rollout,
            cpuct: cpuct,
            temperature: temperature,
        }
    }
}

impl<G: game::Game<chess::BoardState>> Player for NNPlayer<G> {
    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize> {
        mcts::mcts(
            &self.game,
            cursor.current(),
            &state,
            self.n_rollout,
            Some(self.cpuct),
        );

        let num_act_vec: Vec<i32> = cursor
            .current()
            .children
            .iter()
            .map(|a| a.num_act)
            .collect();
        if num_act_vec.len() == 0 {
            return None;
        }

        let choice: usize = if self.temperature == 0.0 {
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
            let power = 1.0 / self.temperature;
            let weights =
                WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
            weights.sample(&mut thread_rng())
        };

        Some(choice)
    }
}

fn step(choice: usize, cursor: &mut MctsCursor, state: &mut chess::BoardState, trace: &mut Trace) {
    let q_value = cursor.current().q_value;
    let all_children: Vec<(chess::Move, i32, f32)> = cursor
        .current()
        .children
        .iter()
        .map(|c| (c.step.0.unwrap(), c.num_act, c.q_value))
        .collect();

    cursor.move_children(choice);
    let step = cursor.current().step;
    let mov = step.0.unwrap();
    let turn = !step.1;
    println!("{}: {}", turn, mov);
    trace.push(Some(mov), q_value, all_children);
    state.next(&mov);
}

fn play_loop<W, B>(
    white: &W,
    black: &B,
    cursor: &mut MctsCursor,
    state: &mut chess::BoardState,
    trace: &mut Trace,
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
        Some("onnx") => Box::new(NNPlayer::<game::ChessOnnx>::load(
            device,
            path,
            n_rollout,
            cpuct,
            temperature,
        )) as Box<SomeNNPlayer>,
        Some("pt") => Box::new(NNPlayer::<game::ChessTS>::load(
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
        match libloading::Library::new("libtorchtrt.so") {
            Err(e) => println!("torch_tensorrt not found: {}", e),
            Ok(_) => (),
        }
    }
    let args = Args::parse();
    let _ = create_dir("replay");

    let mut trace = trace::Trace::new();
    let mut state = chess::BoardState::new();
    let mut root = mcts::Node {
        step: (None, chess::Color::White),
        depth: 0,
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    };
    let mut cursor = root.as_cursor_mut();

    let white_checkpoint = Path::new(&args.white_checkpoint);
    let white = load_checkpoint(
        white_checkpoint,
        &args.white_device,
        args.rollout,
        args.cpuct,
        args.temperature,
    );

    match args.black_type {
        Opponent::Stockfish => {
            let black = StockfishPlayer::load(&args.stockfish_bin, args.stockfish_level);
            println!("Players loaded.");
            play_loop::<SomeNNPlayer, StockfishPlayer>(
                white.borrow(),
                &black,
                &mut cursor,
                &mut state,
                &mut trace,
            );
        }
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
    trace.save(&(String::from("replay/") + &args.output));
}
