use std::ptr::NonNull;
use pyo3::prelude::*;
use pyo3::intern;
use pyo3::types::PyTuple;
use numpy::array::{PyArray1, PyArray3};
use crate::mcts::Node;
use crate::chess::{Board, BoardState, BoardHistory, Move, Color};

const LOOKBACK: usize = 8;

pub trait Game<S> where S: State {
    fn predict(&self, node: &Node<S::Step>, state: &S, argmax: bool)
        -> (Vec<S::Step>, Vec<f32>, f32);
}

pub trait State {
    type Step;
    fn dup(&self) -> Self;
    fn advance(&mut self, step: &Self::Step);
}

struct GameWithNN {
    model: Py<PyAny>,
}


impl Game<BoardState> for GameWithNN {
    fn predict(&self, node: &Node<Board>, state: &BoardState, argmax: bool) -> (Vec<Board>, Vec<f32>, f32) {
        let mut history = BoardHistory::new(LOOKBACK);

        let mut cur = NonNull::from(node);
        for step in 0..LOOKBACK {
            let ref n = cur.as_ref();
            history.push(&n.step);
            match n.parent {
                None => break,
                Some(parent) => cur = parent,
            }
        }

        let next_steps = state.next_steps();

        let rotate = node.step.turn == Color::Black;
        let encoded_boards = history.view(rotate);
        let encoded_meta = node.step.encode_meta();

        let (moves_distr, score) = Python::with_gil(|py| {
            let encoded_boards = PyArray3::from_array(py, &encoded_boards);
            let encoded_meta = PyArray3::from_array(py, &encoded_meta);
            let output = self.model.call_method1(
                py, intern!(py, "forward"),
                (encoded_boards,encoded_meta)
            ).unwrap();
            let tuple: &PyTuple = output.downcast(py).unwrap();
            let full_distr: &PyArray1<f32> = tuple.get_item(0).and_then(|o| o.extract()) .unwrap();
            let score: f32 = tuple.get_item(1).and_then(|o| o.extract()).unwrap();

            let encoded_moves: Vec<i32> = next_steps
                .iter()
                .map(|b| {
                    let m = b.last_move.unwrap();
                     if rotate {m.rotate().encode()} else {m.encode()}
                })
                .collect();
            let moves_distr: Vec<f32> = full_distr
                .readonly()
                .get_item(encoded_moves.to_object(py))
                .and_then(|o| o.extract())
                .unwrap();

            return (moves_distr, score);
        });

        let moves_distr: Vec<f32> = if argmax {
            let i: usize = moves_distr.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            let mut moves_distr = vec![0.; moves_distr.len()];
            moves_distr[i] = 1.;
            moves_distr
        }
        else {
            let sum = moves_distr.iter().sum::<f32>() + 1e-5;
            moves_distr.iter().map(|x| x / sum).collect()
        };

        return (next_steps, moves_distr, score);
    }
}
