use pyo3::prelude::*;

mod chess;
mod knightmoves;
mod underpromotions;
mod queenmoves;
mod mcts;
mod game;


fn get_python_path() -> Py<PyAny> {
    return Python::with_gil(|py| {
        py.import("sys").and_then(|m| m.getattr("path")).unwrap().into()
    });
}

fn print_node(node: &mcts::Node<chess::Board>) {
    for (idx, ch) in node.children.iter().enumerate() {
        println!("Child {}: {} {} {}", idx, ch.q_value, ch.num_act, ch.step.last_move.unwrap());
    }
}

fn step(cursor: &mut mcts::CursorMut<chess::Board>, state: &mut chess::BoardState) -> bool {
    let node = cursor.current();
    let opt_choice = node.children.iter().enumerate().max_by(|a, b| {
        let v1 = a.1.q_value / a.1.num_act as f32;
        let v2 = b.1.q_value / b.1.num_act as f32;
        v1.partial_cmp(&v2).unwrap()
    });
    match opt_choice {
        None => return false;
        Some((idx, _)) => {
            cursor.move_children(idx);
            game::State::advance(state, &cursor.current().step);
            return true;
        }
    }
}

fn main() {
    let r = Python::with_gil(|py| {
        let nn = py.import("nn")?.getattr("load_model")?.call0()?;
        Ok::<Py<PyAny>, PyErr>(nn.into())
    });


    match r {
        Ok(nn) => {
            let chess = game::Chess{model: nn};
            let mut state = chess::BoardState::new();
            let mut root = mcts::Node {
                step: state.to_board(),
                q_value: 0.,
                num_act: 0,
                parent: None,
                children: Vec::new()
            };
            let mut cursor = root.as_cursor_mut();

            for i in 0..5 {
                mcts::mcts(&chess, cursor.current(), &state, 100, false, None);
                if !step(&mut cursor, &mut state) {
                    break;
                }
                println!("Step {}\n{}", i, state);
            }
            //mcts::mcts(&chess, &mut n1, &state, 100, true, None);
            //println!("Current: {} {} {}", n1.q_value, n1.num_act, n1.step.turn);
            //print_node(n1);

            //let mut n2 = select(&mut n1);
            //println!("next: {} {}", n2.step.last_move.unwrap(), n2.step.turn);
            //game::State::advance(&mut state, &n2.step);
            //println!("{}", state);
        }
        Err(e) => {
            let path = get_python_path();
            println!("Error:\n {}\n when importing python module 'nn' from path: {}", e, path);
        }
    }
}
