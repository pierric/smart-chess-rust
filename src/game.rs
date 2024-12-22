use crate::chess::{get_board_from_moves, BoardHistory, BoardState, Color, Move};
use crate::mcts::Node;
use cached::proc_macro::cached;
use cached::SizedCache;
use ndarray::Array3;
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyString};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

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
    fn legal_moves(&self) -> Vec<Self::Step>;
}

pub struct Chess {
    pub model: Py<PyAny>,
    pub device: String,
}

pub struct ChessTS {
    pub model: tch::CModule,
    pub device: tch::Device,
}

/*
pub struct ChessOnnx {
    pub session: ort::Session,
}
*/

#[cached(
    type = "SizedCache<(String, Color, u64), (Vec<f32>, f32)>",
    create = "{ SizedCache::with_size(5000) }",
    convert = r#"{
        let mut hasher = DefaultHasher::new();
        boards.hash(&mut hasher);
        meta.hash(&mut hasher);
        steps.hash(&mut hasher);
        (String::from(device), turn, hasher.finish())
    }"#
)]
fn call_py_model(
    model: &Py<PyAny>,
    device: &str,
    boards: Array3<i32>,
    meta: Array3<i32>,
    turn: Color,
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
        // how good is the current board for the next player (turn)
        // This must be in sync with that in training script
        let score = score * (if turn == Color::Black { -1. } else { 1. });

        // rotate if the next move is black
        let encoded_moves: Vec<i32> = steps
            .iter()
            .map(|m| {
                if turn == Color::Black {
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

fn _encode(node: &Node<(Option<Move>, Color)>, state: &BoardState) -> (Array3<i32>, Array3<i32>) {
    let current_board = state.to_board();
    let mut history = BoardHistory::new(LOOKBACK);
    let mut move_stack = state.move_stack();

    let mut cur = NonNull::from(node);
    for _ in 0..LOOKBACK {
        let n = unsafe { cur.as_ref() };
        // latter boards are kept at the front
        // must be in sync with encode_steps in lib.rs
        history.push_back(&get_board_from_moves(&move_stack));
        match n.parent {
            None => break,
            Some(parent) => {
                let m = move_stack.pop();
                assert!(m == n.step.0, "sanity check on the last move failed");
                cur = parent;
            }
        }
    }

    let encoded_boards = history.view(node.step.1 == Color::Black);
    let encoded_meta = current_board.encode_meta();

    (encoded_boards, encoded_meta)
}

fn _post_process_distr(distr: Vec<f32>, argmax: bool) -> Vec<f32> {
    if argmax {
        let i: usize = distr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let mut distr = vec![0.; distr.len()];
        distr[i] = 1.;
        distr
    } else {
        let sum = distr.iter().sum::<f32>() + 1e-5;

        if !sum.is_finite() {
            println!("Warning: {:?}", distr);
        }

        distr.iter().map(|x| x / sum).collect()
    }
}

#[cached(
    type = "SizedCache<(bool, Vec<Move>, Color), (Vec<(Option<Move>, Color)>, Vec<f32>, f32)>",
    create = "{ SizedCache::with_size(5000) }",
    convert = r#"{
        (argmax, state.move_stack(), node.step.1)
    }"#
)]
fn _chess_predict(
    chess: &Chess,
    node: &Node<(Option<Move>, Color)>,
    state: &BoardState,
    argmax: bool,
) -> (Vec<(Option<Move>, Color)>, Vec<f32>, f32) {
    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(node, state);

    let (moves_distr, score) = call_py_model(
        &chess.model,
        &chess.device,
        encoded_boards,
        encoded_meta,
        node.step.1,
        &legal_moves,
    );

    let moves_distr = _post_process_distr(moves_distr, argmax);
    let next_steps = legal_moves
        .into_iter()
        .map(|m| (Some(m), !node.step.1))
        .collect();
    return (next_steps, moves_distr, score);
}

impl Game<BoardState> for Chess {
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

fn call_ts_model(
    model: &tch::CModule,
    device: tch::Device,
    boards: Array3<i32>,
    meta: Array3<i32>,
    turn: Color,
    steps: &Vec<Move>,
) -> (Vec<f32>, f32) {
    use tch::Tensor;

    let encoded_boards =
        Tensor::try_from(boards)
            .unwrap()
            .to_device_(device, tch::Kind::Float, true, false);
    let encoded_meta =
        Tensor::try_from(meta)
            .unwrap()
            .to_device_(device, tch::Kind::Float, true, false);

    let inp = Tensor::cat(&[encoded_boards, encoded_meta], 2)
        .permute([2, 0, 1])
        .unsqueeze(0);
    let out = model.forward_is(&[tch::jit::IValue::from(inp)]).unwrap();
    let (full_distr, score) = <(Tensor, Tensor)>::try_from(out).unwrap();

    // how good is the current board for the next player (turn)
    // This must be in sync with that in training script
    let score = score.to_dtype(tch::Kind::Float, true, false);
    let score = f32::try_from(&score).unwrap() * (if turn == Color::Black { -1. } else { 1. });

    // rotate if the next move is black
    let encoded_moves: Vec<i64> = steps
        .iter()
        .map(|m| {
            if turn == Color::Black {
                m.rotate().encode() as i64
            } else {
                m.encode() as i64
            }
        })
        .collect();
    let encoded_moves = Tensor::from_slice(&encoded_moves).to_device(device);
    let moves_distr = Vec::<f32>::try_from(full_distr.take(&encoded_moves)).unwrap();

    (moves_distr, score)
}

#[cached(
    type = "SizedCache<(bool, Vec<Move>, Color), (Vec<(Option<Move>, Color)>, Vec<f32>, f32)>",
    create = "{ SizedCache::with_size(10000) }",
    convert = r#"{
        (argmax, state.move_stack(), node.step.1)
    }"#
)]
fn _chess_ts_predict(
    chess: &ChessTS,
    node: &Node<(Option<Move>, Color)>,
    state: &BoardState,
    argmax: bool,
) -> (Vec<(Option<Move>, Color)>, Vec<f32>, f32) {
    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(node, state);

    let (moves_distr, score) = call_ts_model(
        &chess.model,
        chess.device,
        encoded_boards,
        encoded_meta,
        node.step.1,
        &legal_moves,
    );

    let moves_distr = _post_process_distr(moves_distr, argmax);
    let next_steps = legal_moves
        .into_iter()
        .map(|m| (Some(m), !node.step.1))
        .collect();
    return (next_steps, moves_distr, score);
}

impl Game<BoardState> for ChessTS {
    fn predict(
        &self,
        node: &Node<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        _chess_ts_predict(self, node, state, argmax)
    }

    fn reverse_q(&self, node: &Node<<BoardState as State>::Step>) -> bool {
        node.step.1 == Color::Black
    }
}

/*
fn call_onnx_model(
    session: &ort::Session,
    boards: Array3<i32>,
    meta: Array3<i32>,
    turn: Color,
    steps: &Vec<Move>,
) -> ort::Result<(Vec<f32>, f32)> {
    let cat = concatenate![Axis(2), boards, meta].mapv(|v| v as f32);
    let inp = cat.view().permuted_axes([2, 0, 1]).insert_axis(Axis(0));
    let out = session.run(ort::inputs!["inp" => inp]?)?;

    let full_distr: ort::TensorRef<f32> = out[0].downcast_ref()?;
    let score: ort::TensorRef<f32> = out[1].downcast_ref()?;
    let score = score.index([0, 0]) * (if turn == Color::Black { -1. } else { 1. });

    // rotate if the next move is black
    let moves_distr: Vec<f32> = steps
        .iter()
        .map(|m| {
            let mi = if turn == Color::Black {
                m.rotate().encode() as i64
            } else {
                m.encode() as i64
            };
            *full_distr.index([0, mi])
        })
        .collect();

    Ok((moves_distr, score))
}

#[cached(
    type = "SizedCache<(bool, Vec<Move>, Color), (Vec<(Option<Move>, Color)>, Vec<f32>, f32)>",
    create = "{ SizedCache::with_size(10000) }",
    convert = r#"{
        (argmax, state.move_stack(), node.step.1)
    }"#
)]
fn _chess_onnx_predict(
    chess: &ChessOnnx,
    node: &Node<(Option<Move>, Color)>,
    state: &BoardState,
    argmax: bool,
) -> (Vec<(Option<Move>, Color)>, Vec<f32>, f32) {
    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(node, state);

    let (moves_distr, score) = call_onnx_model(
        &chess.session,
        encoded_boards,
        encoded_meta,
        node.step.1,
        &legal_moves,
    )
    .unwrap();

    let moves_distr = _post_process_distr(moves_distr, argmax);
    let next_steps = legal_moves
        .into_iter()
        .map(|m| (Some(m), !node.step.1))
        .collect();
    return (next_steps, moves_distr, score);
}

impl Game<BoardState> for ChessOnnx {
    fn predict(
        &self,
        node: &Node<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        _chess_onnx_predict(self, node, state, argmax)
    }

    fn reverse_q(&self, node: &Node<<BoardState as State>::Step>) -> bool {
        node.step.1 == Color::Black
    }
}
*/
