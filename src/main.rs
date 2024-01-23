use clap::Parser;
use pyo3::prelude::*;

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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    device: String,

    #[arg(short, long, default_missing_value = None)]
    rollout_factor: Option<f32>,

    #[arg(long, default_missing_value = None)]
    rollout_num: Option<i32>,

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
    let args = Args::parse();

    assert!(args.rollout_factor.is_none() || args.rollout_num.is_none());

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
                device: args.device.clone(),
            };

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
                let temperature = if i < 8 {1.0} else {args.temperature};

                let rollout = match (args.rollout_factor, args.rollout_num) {
                    (Some(v), None) => i32::max(200, (state.legal_moves().len() as f32 * v) as i32),
                    (None, Some(v)) => v,
                    (None, None) => 300,
                    (Some(_), Some(_)) => panic!("both --rollout_factor and --rollout_num are specified."),
                };
                println!("Rollout {} Temp {} Cpuct {} Turn {}", rollout, temperature, args.cpuct, cursor.current().step.1);
                mcts::mcts(&chess, cursor.current(), &state, rollout, Some(args.cpuct));

                let node = cursor.current();
                let q_value = node.q_value;
                let num_act_children: Vec<(chess::Move, i32, f32)> = node
                    .children
                    .iter()
                    .map(|c| (c.step.0.unwrap(), c.num_act, c.q_value))
                    .collect();


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
        Err(e) => {
            let path = get_python_path();
            println!(
                "Error:\n {}\n when importing python module 'nn' from path: {}",
                e, path
            );
        }
    }
}
