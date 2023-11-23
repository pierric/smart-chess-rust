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

fn select(node: &mut mcts::Node<chess::Board>) -> &mut mcts::Node<chess::Board> {
    node.children.iter_mut().max_by(|a, b| {
        let v1 = a.q_value / a.num_act as f32;
        let v2 = b.q_value / b.num_act as f32;
        v1.partial_cmp(&v2).unwrap()
    }).unwrap().as_mut()
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
            let board = state.to_board();
            let mut root = mcts::Node{step: board, q_value: 0., num_act: 0, parent: None, children: Vec::new()};

            mcts::mcts(&chess, &mut root, &state, 100, false, None);
            println!("Root: {} {} {}", root.q_value, root.num_act, root.step.turn);
            //print_node(&root);

            let mut n1 = select(&mut root);
            println!("next: {} {}", n1.step.last_move.unwrap(), n1.step.turn);
            game::State::advance(&mut state, &n1.step);
            println!("{}", state);

            mcts::mcts(&chess, &mut n1, &state, 100, true, None);
            println!("Current: {} {} {}", n1.q_value, n1.num_act, n1.step.turn);
            //print_node(n1);

            let mut n2 = select(&mut n1);
            println!("next: {} {}", n2.step.last_move.unwrap(), n2.step.turn);
            game::State::advance(&mut state, &n2.step);
            println!("{}", state);

            mcts::mcts(&chess, &mut n2, &state, 100, false, None);
            println!("Current: {} {} {}", n2.q_value, n2.num_act, n2.step.turn);
            //print_node(n2);

            let mut n3 = select(&mut n2);
            println!("next: {} {}", n3.step.last_move.unwrap(), n3.step.turn);
            game::State::advance(&mut state, &n3.step);
            println!("{}", state);


        }
        Err(e) => {
            let path = get_python_path();
            println!("Error:\n {}\n when importing python module 'nn' from path: {}", e, path);
        }
    }
}
