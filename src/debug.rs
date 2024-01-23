use clap::Parser;
use std::fs::File;
use pyo3::prelude::*;

mod mcts;
mod game;
mod chess;
mod underpromotions;
mod queenmoves;
mod knightmoves;
mod trace;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 300)]
    rollout_num: i32,

    #[arg(short, long, default_value_t = String::from("trace.json"))]
    trace_file: String,

    #[arg(short, long, default_missing_value = None)]
    checkpoint: Option<String>,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 3.0)]
    cpuct: f32,
}

#[allow(unused_variables, dead_code, unused_mut)]
fn debug_step(chess: game::Chess, filename: &str, target_step: usize) {
    use std::io::BufReader;
    use std::ptr::NonNull;
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
    for idx in 0..(target_step-1) {
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

    let moves = steps[target_step].as_array().unwrap()[2].as_array().unwrap();
    let current = cursor.current();
    let parent = NonNull::new(current as *mut _);
    current.children.extend(
        moves
        .into_iter()
        .map(|m| {
            let spec = m.as_array().unwrap();
            let uci = spec[0].as_str().unwrap();
            let mov = chess::Move::from_uci(uci);
            let num = spec[1].as_i64().unwrap() as i32;
            let turn = if target_step % 2 == 0 {chess::Color::White} else {chess::Color::Black};
            Box::new(mcts::Node {
                step: (Some(mov), turn),
                q_value: 0.,
                num_act: num,
                parent: parent,
                children: Vec::new(),
            })
        }));    

    let mov = mcts::step(&mut cursor, &mut state, 0.0).unwrap().0;
    println!("Move {:?}", mov.map(|m| m.uci()));
}

#[allow(unused_variables, dead_code, unused_mut)]
fn debug_trace(chess: game::Chess, filename: &str, target_step: usize, args: Args) {
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
    println!("rollout: {} cpuct: {}", args.rollout_num, args.cpuct);

    println!("{:?}", cursor.current().step.1);
    println!("{}", state);

    let (steps, prior, outcome) = chess.predict(cursor.current(), &state, false);

    mcts::mcts(&chess, cursor.current(), &state, args.rollout_num, Some(args.cpuct));

    let move_piror: Vec<(String, f32)> =
        cursor.current().children.iter().zip(prior.iter()).map(|(n,p)| (n.step.0.unwrap().uci(), *p)).collect();
    println!("{:?}", outcome);

    let children_num_act: Vec<(String, i32, f32, f32)> =
       cursor.current().children.iter().map(|n| (n.step.0.unwrap().uci(), n.num_act, n.q_value, n.q_value / n.num_act as f32)).collect();
    
    use string_builder::Builder;
    let mut builder = Builder::default();
    for c in children_num_act.iter().zip(move_piror.iter()) {
        builder.append(format!(" {:?} {:?}\n", c.0, c.1))
    }
    println!("{}", builder.string().unwrap());
}

fn main() {
    // unsafe { backtrace_on_stack_overflow::enable() };
    let args = Args::parse();

    let r = Python::with_gil(|py| {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("device", "cuda")?;
        kwargs.set_item("checkpoint", &args.checkpoint)?;
        let nn: &PyAny = py
            .import("nn")?
            .getattr("load_model")?
            .call((), Some(kwargs))?;
        Ok::<Py<PyAny>, PyErr>(nn.into())
    });

    match r {
        Ok(nn) => {
            let chess = game::Chess {
                model: nn,
                device: String::from("cuda"),
            };

            // from tournament, model #308, play #4
            // let trace = ("eval/4.json", 3);

            // from training/self-play round #12
            let trace = ("eval/trace7.json", 12);

            debug_trace(chess, trace.0, trace.1, args);
            // debug_step(chess, "runs/86/trace8.json", 9);
        },
        Err(_) => todo!()
    }
}