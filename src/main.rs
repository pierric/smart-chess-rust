use clap::Parser;
use pyo3::prelude::*;
use std::fs::File;
use rand::thread_rng;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

mod chess;
mod game;
mod knightmoves;
mod mcts;
mod queenmoves;
mod underpromotions;
mod trace;

fn get_python_path() -> Py<PyAny> {
    return Python::with_gil(|py| {
        py.import("sys")
            .and_then(|m| m.getattr("path"))
            .unwrap()
            .into()
    });
}

fn step(
    cursor: &mut mcts::CursorMut<<chess::BoardState as game::State>::Step>,
    state: &mut chess::BoardState,
    temp: f32,
) -> Option<chess::Move> {
    let num_act_vec = cursor.current().children.iter().map(|a| a.num_act);

    if num_act_vec.len() == 0 {
        return None
    }

    let choice: usize = if temp == 0.0 {
        num_act_vec.enumerate().max_by(|a, b| {
            a.1.cmp(&b.1)
        }).unwrap().0
    } else {
        let power = 1.0 / temp;
        let weights = WeightedIndex::new(num_act_vec.map(|n| (n as f32).powf(power))).unwrap();
        weights.sample(&mut thread_rng())
    };

    cursor.move_children(choice);
    let step = &cursor.current().step;
    game::State::advance(state, step);
    step.0
}

#[allow(unused_variables, dead_code, unused_mut)]
fn debug_trace(chess: game::Chess, filename: &str, target_step: usize) {
    use std::io::BufReader;
    use std::ptr::NonNull;
    use crate::game::Game;
    let mut file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let trace: serde_json::Map<String, serde_json::Value> = serde_json::from_reader(reader).unwrap();
    let steps = trace["steps"].as_array().unwrap();

    let mut state = chess::BoardState::new();
    let mut root = mcts::Node {
        step: (None, chess::Color::White),
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    };
    let mut cursor = root.as_cursor_mut();
    for idx in 0..target_step {
        let mov_uci = steps[idx].as_array().unwrap()[0].as_str().unwrap();
        let mov = chess::Move::from_uci(mov_uci);
        let legal_moves = state.legal_moves();
        let choice = legal_moves.iter().position(|m| *m == mov).unwrap();
        let current = cursor.current();
        let turn = current.step.1;
        let parent = NonNull::new(current as *mut _);
        current.children.extend(
            legal_moves
            .into_iter()
            .map(|m| {
                Box::new(mcts::Node {
                    step: (Some(m), !turn),
                    q_value: 0.,
                    num_act: 0,
                    parent: parent,
                    children: Vec::new(),
                })
            }));
        cursor.move_children(choice);
        game::State::advance(&mut state, &cursor.current().step);
    }

    //let rollout = (state.legal_moves().len() as f32 * 6.0) as i32;
    let rollout = 160;
    println!("rollout: {}", rollout);

    println!("{:?}", cursor.current().step.1);
    println!("{}", state);

    let (steps, prior, outcome) = chess.predict(cursor.current(), &state, false);
    println!("{:?} {:?}", outcome, prior);

    mcts::mcts(&chess, cursor.current(), &state, rollout, Some(0.05));
    let children_num_act: Vec<(String, i32, f32)> =
       cursor.current().children.iter().map(|n| (n.step.0.unwrap().uci(), n.num_act, n.q_value)).collect();
    println!("{:?}", children_num_act);
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    device: String,

    #[arg(short, long, default_value_t = 2.0)]
    rollout_factor: f32,

    #[arg(short, long, default_value_t = 10)]
    num_steps: i32,

    #[arg(short, long, default_value_t = String::from("trace.json"))]
    trace_file: String,

    #[arg(short, long, default_missing_value = None)]
    checkpoint: Option<String>,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    cpuct: f32,
}

fn main() {
    unsafe { backtrace_on_stack_overflow::enable() };
    let args = Args::parse();

    let r = Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("device", &args.device)?;
        kwargs.set_item("checkpoint", &args.checkpoint)?;
        let nn: &PyAny = py
            .import("nn")?
            .getattr("load_model")?
            .call((), Some(kwargs))?;
        Ok::<Py<PyAny>, PyErr>(nn.into())
    });

    match r {
        Ok(nn) => {
            let mut trace = trace::Trace::new();
            let chess = game::Chess {
                model: nn,
                device: &args.device,
            };

            // debug_trace(chess, "debug-traces/trace4.json", 5);
            // return;

            let mut state = chess::BoardState::new();
            let mut root = mcts::Node {
                step: (None, chess::Color::White),
                q_value: 0.,
                num_act: 0,
                parent: None,
                children: Vec::new(),
            };
            let mut cursor = root.as_cursor_mut();
            let mut outcome = None;

            for i in 0..args.num_steps {
                let temperature = if i < 10 {1.0} else {args.temperature};
                let rollout = i32::max(150, (state.legal_moves().len() as f32 * args.rollout_factor) as i32);

                println!("Rollout {} Temp {} Cpuct {} Turn {}", rollout, temperature, args.cpuct, cursor.current().step.1);
                mcts::mcts(&chess, cursor.current(), &state, rollout, Some(args.cpuct));

                let node = cursor.current();
                let q_value = node.q_value;
                let num_act_children: Vec<(chess::Move, i32, f32)> = node
                    .children
                    .iter()
                    .map(|c| (c.step.0.unwrap(), c.num_act, c.q_value))
                    .collect();


                match step(&mut cursor, &mut state, temperature) {
                    None => {
                        outcome = state.outcome();
                        break;
                    }
                    mov => {
                        trace.push(mov, q_value, num_act_children);
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
        Err(e) => {
            let path = get_python_path();
            println!(
                "Error:\n {}\n when importing python module 'nn' from path: {}",
                e, path
            );
        }
    }
}
