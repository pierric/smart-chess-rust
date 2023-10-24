use pyo3::prelude::{Py, PyAny};
use crate::mcts::Node;
use crate::chess::{Board, BoardHistory};

pub trait Game<T> {
    fn predict(&self, node: &Node<T>, argmax: bool) -> (Vec<T>, Vec<f32>, f32);
}

struct GameWithNN {
    model: Py<PyAny>,
    history: BoardHistory,
}

impl<'a> Game<Board<'a>> for GameWithNN {
    fn predict(&self, node: &Node<Board<'a>>, argmax: bool) -> (Vec<Board<'a>>, Vec<f32>, f32) {
        let &board = node.step;
        
    }
}
