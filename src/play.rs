use std::ptr::NonNull;
use pyo3::prelude::*;
use clap::Parser;
use uci::Engine;
use rand::thread_rng;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

mod chess;
mod knightmoves;
mod queenmoves;
mod underpromotions;
mod mcts;
mod game;
mod trace;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    device: String,

    #[arg(short, long, default_value = "/usr/bin/stockfish")]
    stockfish: String,

    #[arg(short, long, default_value_t = 0)]
    level: i32,

    #[arg(short, long, default_value_t = 60)]
    rollout: i32,

    #[arg(short, long)]
    checkpoint: String,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 0.0)]
    cpuct: f32,
}

fn main() {
    unsafe { backtrace_on_stack_overflow::enable() };
    let args = Args::parse();

    let nn = Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("device", &args.device)?;
        kwargs.set_item("checkpoint", &args.checkpoint)?;
        let nn: &PyAny = py
            .import("nn")?
            .getattr("load_model")?
            .call((), Some(kwargs))?;
        Ok::<Py<PyAny>, PyErr>(nn.into())
    }).unwrap();
    println!("Model loaded.");

    let mut trace = trace::Trace::new();

    let stockfish = Engine::new(&args.stockfish).unwrap();
    stockfish.set_option("Skill Level", &args.level.to_string()).unwrap();

    let chess = game::Chess { model: nn, device: &args.device };

    let mut state = chess::BoardState::new();
    let mut root = mcts::Node {
        step: (None, chess::Color::White),
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    };
    let mut cursor = root.as_cursor_mut();

    loop {
        mcts::mcts(&chess, cursor.current(), &state, args.rollout, Some(args.cpuct));

        let num_act_vec: Vec<i32> = cursor.current().children.iter().map(|a| a.num_act).collect();
        if num_act_vec.len() == 0 {
            break;
        }

        let choice: usize = if args.temperature == 0.0 {
            num_act_vec.iter().enumerate().max_by(|a, b| {a.1.cmp(&b.1)}).unwrap().0
        } else {
            let power = 1.0 / args.temperature;
            let weights = WeightedIndex::new(num_act_vec.iter().map(|n| (*n as f32).powf(power))).unwrap();
            weights.sample(&mut thread_rng())
        };

        let q_value = cursor.current().q_value;
        let all_children: Vec<(chess::Move, i32, f32)> = cursor.current()
            .children
            .iter()
            .map(|c| (c.step.0.unwrap(), c.num_act, c.q_value))
            .collect();

        cursor.move_children(choice);
        let mov = cursor.current().step.0.unwrap();
        println!("White: {}", mov);
        trace.push(Some(mov), q_value, all_children);
        state.next(&mov);

        stockfish.set_position(&state.fen()).unwrap();
        let mov = chess::Move::from_uci(&stockfish.bestmove().unwrap());
        println!("Black: {}", mov);
        trace.push(Some(mov), 0.0, vec![(mov, 1, 0.0)]);
        state.next(&mov);

        if state.outcome().is_some() {
            break;
        }

        let node = cursor.current();
        let parent = NonNull::new(node as *mut _);
        node.children.clear();
        node.children.push(
            Box::new(mcts::Node {
                step: (Some(mov), chess::Color::White),
                q_value: 0.,
                num_act: 0,
                parent: parent,
                children: Vec::new()
            }));
        cursor.move_children(0);
    }

    let outcome = state.outcome().unwrap();
    println!("{:?}", outcome);
    trace.set_outcome(outcome);
    trace.save("replay.json");
}
