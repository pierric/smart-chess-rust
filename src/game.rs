use std::ptr::NonNull;
use pyo3::prelude::*;
use pyo3::intern;
use pyo3::types::PyTuple;
use numpy::array::{PyArray1, PyArray3};
use crate::mcts::Node;
use crate::chess::{Board, BoardHistory, Move, Color};

const LOOKBACK: usize = 8;

pub trait Game<T> {
    fn predict(&self, node: &Node<T>, argmax: bool) -> (Vec<T>, Vec<f32>, f32);
}

struct GameWithNN {
    model: Py<PyAny>,
    history: BoardHistory,
}

impl<'a> Game<Board<'a>> for GameWithNN {
    fn predict(&self, node: &Node<Board<'a>>, argmax: bool) -> (Vec<Board<'a>>, Vec<f32>, f32) {
        let history = BoardHistory::new(LOOKBACK);

        let mut cur = NonNull::from(node);
        for step in 0..LOOKBACK {
            let ref n = cur.as_ref();
            history.push(n.step);
            match n.parent {
                None => break,
                Some(parent) => cur = parent,
            }
        }

        let rotate = node.step.turn == Color::Black;
        let encoded_boards = history.view(rotate);
        let encoded_meta = node.step.encode_meta();

        let (distr, score) = Python::with_gil(|py| {
            let encoded_boards = PyArray3::from_array(py, &encoded_boards).into_py(py);
            let encoded_meta = PyArray3::from_array(py, &encoded_meta).into_py(py);
            let output = self.model.call_method1(
                py, intern!(py, "forward"),
                (encoded_boards,encoded_meta)
            ).unwrap();
            let tuple: &PyTuple = output.downcast(py).unwrap();
            let distribution: &PyArray1<f32> = tuple.get_item(0).and_then(|o| o.extract()) .unwrap();
            let score: f32 = tuple.get_item(1).and_then(|o| o.extract()).unwrap();

            let moves = PyArray1::from_vec(py, node.step.legal_moves());

            (distribution.readonly().as_array(), score);
        });

    }
}
