use clap::Parser;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Write;
use rand::thread_rng;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;

mod chess;
mod game;
mod knightmoves;
mod mcts;
mod queenmoves;
mod underpromotions;

fn get_python_path() -> Py<PyAny> {
    return Python::with_gil(|py| {
        py.import("sys")
            .and_then(|m| m.getattr("path"))
            .unwrap()
            .into()
    });
}

fn step(
    cursor: &mut mcts::CursorMut<chess::Board>,
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
    step.last_move
}

fn save_trace(
    filename: &str,
    trace: Vec<(Option<chess::Move>, f32, Vec<(i32, f32)>)>,
    outcome: Option<chess::Outcome>,
) {
    let json = serde_json::json!({
        "steps": trace,
        "outcome": outcome,
    });

    let mut file = File::create(filename).unwrap();
    file.write_all(json.to_string().as_bytes()).unwrap();
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
            let mut trace: Vec<(Option<chess::Move>, f32, Vec<(i32, f32)>)> = Vec::new();
            let chess = game::Chess {
                model: nn,
                device: &args.device,
            };
            let mut state = chess::BoardState::new();
            let mut root = mcts::Node {
                step: state.to_board(),
                q_value: 0.,
                num_act: 0,
                parent: None,
                children: Vec::new(),
            };
            let mut cursor = root.as_cursor_mut();
            let mut outcome = None;

            for i in 0..args.num_steps {
                let rev = cursor.current().step.turn == chess::Color::Black;
                let temperature = if i < 20 {args.temperature} else {0.5};

                let mut rollout = (state.next_steps().len() as f32 * args.rollout_factor) as i32;
                if i > 200 {
                    rollout = rollout * 2;
                }

                println!("Rollout {:?} Temp {:?}", rollout, temperature);
                mcts::mcts(&chess, cursor.current(), &state, rollout, rev, Some(args.cpuct));

                let node = cursor.current();
                let q_value = node.q_value;
                let num_act_children: Vec<(i32, f32)> = node
                    .children
                    .iter()
                    .map(|c| (c.num_act, c.q_value))
                    .collect();


                match step(&mut cursor, &mut state, temperature) {
                    None => {
                        outcome = state.outcome();
                        break;
                    }
                    mov => {
                        trace.push((mov, q_value, num_act_children));
                        println!("Step {}\n{}", i, state);
                    }
                }

                if i > 150 {
                    outcome = state.outcome();
                    if outcome.is_some() {
                        break;
                    }
                }
            }

            save_trace(&args.trace_file, trace, outcome);
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
