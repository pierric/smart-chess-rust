use crate::chess::{BoardHistory, BoardState, Color, Move};
use crate::mcts::Node;
use ndarray::Array3;
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyString};
use std::ptr::NonNull;
use cached::proc_macro::cached;
use cached::SizedCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const LOOKBACK: usize = 8;

pub trait Game<S>
where
    S: State,
{
    fn predict(
        &self,
        node: &Node<S::Step>,
        state: &S,
        argmax: bool,
    ) -> (Vec<S::Step>, Vec<f32>, f32);

    fn reverse_q(&self, node: &Node<S::Step>) -> bool;
}

pub trait State {
    type Step;
    fn dup(&self) -> Self;
    fn advance(&mut self, step: &Self::Step);
}

pub struct Chess<'a> {
    pub model: Py<PyAny>,
    pub device: &'a str,
}

#[cached(
    type = "SizedCache<(String, bool, u64), (Vec<f32>, f32)>",
    create = "{ SizedCache::with_size(5000) }",
    convert = r#"{
        let mut hasher = DefaultHasher::new();
        boards.hash(&mut hasher);
        meta.hash(&mut hasher);
        steps.hash(&mut hasher);
        (String::from(device), rotate, hasher.finish())
    }"#
)]
fn call_py_model(
    model: &Py<PyAny>,
    device: &str,
    boards: Array3<u32>,
    meta: Array3<u32>,
    rotate: bool,
    steps: &Vec<Move>,
) -> (Vec<f32>, f32) {
    Python::with_gil(|py| {
        let encoded_boards = PyArray3::from_array(py, &boards);
        let encoded_meta = PyArray3::from_array(py, &meta);

        let code = r#"
inp = np.concatenate((encoded_boards, encoded_meta), axis=-1).astype(np.float32)
inp = inp.transpose((2, 0, 1))
inp = torch.from_numpy(inp[np.newaxis, :])
with torch.no_grad():
    ret_distr, ret_score = model(inp.to(device))
ret_distr = ret_distr.detach().cpu().numpy().squeeze()
ret_distr = np.exp(ret_distr)
ret_score = ret_score.detach().cpu().item()
"#;
        let locals = [
            ("np", py.import("numpy").unwrap().as_ref()),
            ("torch", py.import("torch").unwrap().as_ref()),
            ("encoded_boards", encoded_boards.as_ref()),
            ("encoded_meta", encoded_meta.as_ref()),
            ("model", model.as_ref(py)),
            ("device", PyString::new(py, device)),
        ]
        .into_py_dict(py);
        py.run(code, Some(locals), None).unwrap();

        let full_distr: &PyArray1<f32> = locals
            .get_item("ret_distr")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let score: f32 = locals
            .get_item("ret_score")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        // how possible is white going to win
        let score = score * (if rotate { -1. } else { 1. });

        let encoded_moves: Vec<i32> = steps
            .iter()
            .map(|m| {
                if rotate {
                    m.rotate().encode()
                } else {
                    m.encode()
                }
            })
            .collect();
        let moves_distr: Vec<f32> = full_distr
            .readonly()
            .get_item(encoded_moves.to_object(py))
            .and_then(|o| o.extract())
            .unwrap();

        (moves_distr, score)
    })
}

// #[cached(
//     type = "SizedCache<(u64), (Vec<Board>, Vec<f32>, f32)>",
//     create = "{ SizedCache::with_size(50000) }",
//     convert = r#"{
//         let mut hasher = DefaultHasher::new();
//         let ptr = node as *const Node<Board>;
//         ptr.hash(&mut hasher);
//         hasher.finish()
//     }"#
// )]
fn _chess_predict(chess: &Chess, node: &Node<(Option<Move>, Color)>, state: &BoardState, argmax: bool) -> (Vec<(Option<Move>, Color)>, Vec<f32>, f32) {
    let current_board = state.to_board();
    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let mut history = BoardHistory::new(LOOKBACK);
    let mut dup_state = state.dup();

    let mut cur = NonNull::from(node);
    for _ in 0..LOOKBACK {
        let n = unsafe { cur.as_ref() };
        history.push(&dup_state.to_board());
        match n.parent {
            None => break,
            Some(parent) => {
                let m = dup_state.prev();
                assert!(m == n.step.0, "sanity check on the last move failed");
                cur = parent;
            },
        }
    }

    let rotate = node.step.1 == Color::Black;
    let encoded_boards = history.view(rotate);
    let encoded_meta = current_board.encode_meta();

    let (moves_distr, score) = call_py_model(
        &chess.model,
        chess.device,
        encoded_boards,
        encoded_meta,
        rotate,
        &legal_moves,
    );

    let moves_distr: Vec<f32> = if argmax {
        let i: usize = moves_distr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let mut moves_distr = vec![0.; moves_distr.len()];
        moves_distr[i] = 1.;
        moves_distr
    } else {
        let sum = moves_distr.iter().sum::<f32>() + 1e-5;

        if !sum.is_finite() {
            println!("Warning: {:?}", moves_distr);
        }

        moves_distr.iter().map(|x| x / sum).collect()
    };

    let next_steps = legal_moves.into_iter().map(|m| (Some(m), !node.step.1)).collect();
    return (next_steps, moves_distr, score);
}

impl Game<BoardState> for Chess<'_> {
    fn predict(
        &self,
        node: &Node<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        _chess_predict(self, node, state, argmax)
    }

    fn reverse_q(&self, node: &Node<<BoardState as State>::Step>) -> bool {
        node.step.1 == Color::Black
    }
}
