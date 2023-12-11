use clap::Parser;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Write;

mod chess;
mod game;
mod knightmoves;
mod mcts;
mod queenmoves;
mod underpromotions;

const EPSILON: f32 = 1e05;

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
) -> Option<chess::Move> {
    let opt_choice = cursor.current().children.iter().enumerate().max_by(|a, b| {
        let v1 = a.1.q_value / (a.1.num_act as f32 + EPSILON);
        let v2 = b.1.q_value / (b.1.num_act as f32 + EPSILON);
        v1.partial_cmp(&v2).unwrap()
    });
    match opt_choice {
        None => None,
        Some((idx, _)) => {
            cursor.move_children(idx);
            let step = &cursor.current().step;
            game::State::advance(state, step);
            step.last_move
        }
    }
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

    #[arg(short, long, default_value_t = 100)]
    rollout: i32,

    #[arg(short, long, default_value_t = 10)]
    num_steps: i32,

    #[arg(short, long, default_value_t = String::from("trace.json"))]
    trace_file: String,

    #[arg(short, long, default_missing_value = None)]
    checkpoint: Option<String>,
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
                mcts::mcts(&chess, cursor.current(), &state, args.rollout, rev, None);

                let node = cursor.current();
                let q_value = node.q_value;
                let num_act_children: Vec<(i32, f32)> = node
                    .children
                    .iter()
                    .map(|c| (c.num_act, c.q_value))
                    .collect();

                match step(&mut cursor, &mut state) {
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
