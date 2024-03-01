use std::fs::create_dir;
use std::cmp::Eq;
use std::ptr::NonNull;
use clap::{Parser, ValueEnum};
use trace::Trace;
use uci::Engine;
use rand::{thread_rng, Rng};
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

mod chess;
mod knightmoves;
mod queenmoves;
mod underpromotions;
mod mcts;
mod game;
mod trace;

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
    type Config;
    fn load(args: Self::Config) -> Self where Self: Sized;
    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize>;
}

struct StockfishPlayer {
    engine: Engine
}

impl Player for StockfishPlayer {
    type Config = (String, i32);

    fn load(args: Self::Config) -> Self {
        let engine = Engine::new(&args.0).unwrap();
        engine.set_option("Skill Level", &args.1.to_string()).unwrap();
        StockfishPlayer{ engine }
    }

    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize> {
        self.engine.set_position(&state.fen()).unwrap();
        let next = self.engine.bestmove().ok().map(|m| chess::Move::from_uci(&m))?;

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
            node.children.push(
                Box::new(mcts::Node {
                    step: mov,
                    depth: node.depth + 1,
                    q_value: 0.,
                    num_act: num_act,
                    parent: parent,
                    children: Vec::new()
                }));
        }
        assert!(choice.is_some());
        choice
    }
}

struct NNPlayer {
    n_rollout: i32,
    cpuct: f32,
    temperature: f32,
    game: game::ChessTS
}

impl Player for NNPlayer {
    type Config = (String, String, i32, f32, f32);

    fn load(args: Self::Config) -> Self {
        let device = match args.0.as_str() {
            "cpu"  => tch::Device::Cpu,
            "cuda" => tch::Device::Cuda(0),
            _      => todo!("Unsupported device name"),
        };
    
        let chess = game::ChessTS {
            model: tch::CModule::load_on_device(args.1, device).unwrap(),
            device: device,
        };

        NNPlayer {
            game: chess,
            n_rollout: args.2,
            cpuct: args.3,
            temperature: args.4,
        }
    }

    fn bestmove(&self, cursor: &mut MctsCursor, state: &chess::BoardState) -> Option<usize> {
        mcts::mcts(&self.game, cursor.current(), &state, self.n_rollout, Some(self.cpuct));

        let num_act_vec: Vec<i32> = cursor.current().children.iter().map(|a| a.num_act).collect();
        if num_act_vec.len() == 0 {
            return None;
        }

        let choice: usize = if self.temperature == 0.0 {
            let max = num_act_vec.iter().max().unwrap();
            let indices: Vec<usize> = num_act_vec.iter().enumerate().filter(|a| a.1 == max).map(|a| a.0).collect();
            let n: usize = thread_rng().gen_range(0..indices.len());
            indices[n]
        } else {
            let power = 1.0 / self.temperature;
            let weights = WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
            weights.sample(&mut thread_rng())
        };

        Some(choice)
    }
}


fn step(choice: usize, cursor: &mut MctsCursor, state: &mut chess::BoardState, trace: &mut Trace) {
    let q_value = cursor.current().q_value;
    let all_children: Vec<(chess::Move, i32, f32)> = cursor.current()
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

fn play_loop<W, B>(white: W, black: B, cursor: &mut MctsCursor, state: &mut chess::BoardState, trace: &mut Trace) where W: Player, B: Player {
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
            break
        }
    }
}

fn main() {
    unsafe { backtrace_on_stack_overflow::enable() };
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

    let white = NNPlayer::load((args.white_device.clone(), args.white_checkpoint, args.rollout, args.cpuct, args.temperature));

    match args.black_type {
        Opponent::Stockfish => {
            let black = StockfishPlayer::load((args.stockfish_bin, args.stockfish_level));
            println!("Players loaded.");
            play_loop(white, black, &mut cursor, &mut state, &mut trace);    
        },
        Opponent::NN => {
            let black = NNPlayer::load((args.black_device.clone(), args.black_checkpoint, args.rollout, args.cpuct, args.temperature));
            println!("Players loaded.");
            play_loop(white, black, &mut cursor, &mut state, &mut trace);    
        }
    }

    let outcome = state.outcome();
    println!("{:?}", outcome);
    outcome.map(|o| trace.set_outcome(o));
    trace.save(&(String::from("replay/") + &args.output));
}