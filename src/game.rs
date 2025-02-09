use crate::chess::{get_board_from_moves, BoardHistory, BoardState, Color, Move};
use crate::mcts::{Cursor, ArcRefNode};
use ndarray::Array3;
use numpy::{PyArray};
use numpy::array::{PyArray1, PyArray3};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyString};
use short_uuid::ShortUuid;
use tch::Tensor;
use c_str_macro::c_str;

pub const LOOKBACK: usize = 8;

pub trait Game<S>
where
    S: State,
{
    fn predict(
        &self,
        node: &ArcRefNode<S::Step>,
        state: &S,
        argmax: bool,
    ) -> (Vec<S::Step>, Vec<f32>, f32);

    fn reverse_q(&self, node: &ArcRefNode<S::Step>) -> bool;
}

pub trait TchModel {
    fn forward(&self, input: Tensor) -> (Tensor, Tensor);
    fn device(&self) -> tch::Device;
}

pub trait State {
    type Step;
    fn dup(&self) -> Self;
    fn advance(&mut self, step: &Self::Step);
}

pub struct Chess {
    pub model: Py<PyAny>,
    pub device: String,
}

pub struct ChessTS {
    pub model: tch::CModule,
    pub device: tch::Device,
}

pub struct ChessEP {
    pub model: aotinductor::ModelPackage,
    pub device: tch::Device,
}

/*
pub struct ChessOnnx {
    pub session: ort::Session,
}
*/

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

        let code = c_str!(r#"
inp = np.concatenate((encoded_boards, encoded_meta), axis=-1).astype(np.float32)
inp = inp.transpose((2, 0, 1))
inp = torch.from_numpy(inp[np.newaxis, :])
with torch.no_grad():
    with torch.autocast(
        device_type=device, dtype=torch.float16, cache_enabled=False
    ):
        ret_distr, ret_score = model(inp.to(device))
ret_distr = ret_distr.detach().cpu().numpy().squeeze()
ret_distr = np.exp(ret_distr)
ret_score = ret_score.detach().cpu().item()
"#);
        let locals = [
            ("np", py.import("numpy").unwrap().as_ref()),
            ("torch", py.import("torch").unwrap().as_ref()),
            ("encoded_boards", encoded_boards.as_ref()),
            ("encoded_meta", encoded_meta.as_ref()),
            ("model", model.bind(py)),
            ("device", PyString::new(py, device).as_any()),
        ].into_py_dict(py)?;
        py.run(code, None, Some(&locals))?;

        let full_distr: Bound<'_, PyArray1<f32>> = locals
            .get_item("ret_distr")
            .unwrap()
            .unwrap()
            .downcast_into()?;

        // how good is the current board for the next player (turn)
        // This must be in sync with that in training script
        let score: f32 = locals
            .get_item("ret_score")
            .unwrap()
            .unwrap()
            .downcast()?
            .extract()?;

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
            .get_item(PyArray::from_vec(py, encoded_moves))
            .and_then(|o| o.extract())?;

        Ok::<_, PyErr>((moves_distr, score))
    }).unwrap()
}

fn _encode(node: &ArcRefNode<(Option<Move>, Color)>, state: &BoardState) -> (Array3<i32>, Array3<i32>) {
    let current_board = state.to_board();
    let mut history = BoardHistory::new(LOOKBACK);
    let mut move_stack = state.move_stack();

    let mut cursor = Cursor::from_arc(node.clone());
    for _ in 0..LOOKBACK {
        // latter boards are kept at the front
        // must be in sync with encode_steps in lib.rs
        history.push_back(&get_board_from_moves(&move_stack));
        let (parent, last_move) = {
            let n = cursor.current();
            (n.parent.clone(), n.step.0)
        };
        match parent {
            None => break,
            Some(_) => {
                let m = move_stack.pop();
                assert!(m == last_move, "sanity check on the last move failed");
                cursor.navigate_up();
            }
        }
    }

    let turn = node.borrow().step.1;
    let encoded_boards = history.view(turn == Color::Black);
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

        if sum < 0.5 {
            //println!("Warning: move distribution sums up to only {}", sum);
        }

        distr.iter().map(|x| x / sum).collect()
    }
}

fn _chess_predict(
    chess: &Chess,
    node: &ArcRefNode<(Option<Move>, Color)>,
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
    let turn = node.borrow().step.1;

    let (moves_distr, score) = call_py_model(
        &chess.model,
        &chess.device,
        encoded_boards,
        encoded_meta,
        turn,
        &legal_moves,
    );

    let moves_distr = _post_process_distr(moves_distr, argmax);
    let next_steps = legal_moves
        .into_iter()
        .map(|m| (Some(m), !turn))
        .collect();
    return (next_steps, moves_distr, score);
}

impl Game<BoardState> for Chess {
    fn predict(
        &self,
        node: &ArcRefNode<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        _chess_predict(self, node, state, argmax)
    }

    fn reverse_q(&self, node: &ArcRefNode<<BoardState as State>::Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

fn _prepare_tensors(boards: Array3<i32>, meta: Array3<i32>, device: tch::Device) -> Tensor {
    let encoded_boards =
        Tensor::try_from(boards)
            .unwrap()
            .to_device_(device, tch::Kind::BFloat16, true, false);
    let encoded_meta =
        Tensor::try_from(meta)
            .unwrap()
            .to_device_(device, tch::Kind::BFloat16, true, false);

    Tensor::cat(&[encoded_boards, encoded_meta], 2)
        .permute([2, 0, 1])
        .unsqueeze(0)
}

fn _get_score(model_score_output: Tensor) -> f32 {
    // how good is the current board for the next player (turn)
    // This must be in sync with that in training script
    let score = model_score_output.to_dtype(tch::Kind::Float, true, false);
    f32::try_from(&score).unwrap()
}

fn _get_move_distribution(model_distr_output: Tensor, turn: Color, legal_moves: &Vec<Move>, return_full: bool) -> Vec<f32> {
    let full_distr = model_distr_output.to_dtype(tch::Kind::Float, true, false).to_device(tch::Device::Cpu);

    if return_full {
        Vec::<f32>::try_from(full_distr.squeeze().exp()).unwrap()
    }
    else {
        // rotate if the next move is black
        let encoded_moves: Vec<i64> = legal_moves
            .iter()
            .map(|m| {
                if turn == Color::Black {
                    m.rotate().encode() as i64
                } else {
                    m.encode() as i64
                }
            })
            .collect();
        let encoded_moves = Tensor::from_slice(&encoded_moves);
        Vec::<f32>::try_from(full_distr.take(&encoded_moves).exp()).unwrap()
    }
}

pub fn chess_tch_predict<M: TchModel>(
    chess: &M,
    node: &ArcRefNode<(Option<Move>, Color)>,
    state: &BoardState,
    argmax: bool,
    return_full_distr: bool,
) -> (Vec<(Option<Move>, Color)>, Vec<f32>, f32) {

    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        // the model always tell the score at the white side
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(&node, state);
    let turn = node.borrow().step.1;

    assert!(turn == state.turn());

    let inp = _prepare_tensors(encoded_boards, encoded_meta, chess.device());
    let inp_debug = inp.alias_copy();

    let (full_distr, score) = chess.forward(inp);

    let score = _get_score(score);

    if !score.is_finite() {
        let uuid = ShortUuid::generate();
        let filename = format!("/tmp/{}.tensor", uuid);
        inp_debug.save(std::path::Path::new(&filename)).unwrap();
        println!("Warning: score is bad {:?}", score);
        println!("input tensor is saved as {}", filename);
    }

    let moves_distr = _get_move_distribution(full_distr, turn, &legal_moves, return_full_distr);
    let moves_distr = _post_process_distr(moves_distr, argmax);

    let next_steps = legal_moves
        .into_iter()
        .map(|m| (Some(m), !turn))
        .collect();

    return (next_steps, moves_distr, score);
}

impl TchModel for ChessTS {
    fn forward(&self, inp: Tensor) -> (Tensor, Tensor) {
        let out = self.model.forward_is(&[tch::jit::IValue::from(inp)]).unwrap();
        <(Tensor, Tensor)>::try_from(out).unwrap()
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

impl Game<BoardState> for ChessTS {
    fn predict(
        &self,
        node: &ArcRefNode<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<<BoardState as State>::Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

impl TchModel for ChessEP {
    fn forward(&self, inp: Tensor) -> (Tensor, Tensor) {
        let outs = self.model.run(&vec![inp]);
        assert!(outs.len() == 2);
        let mut iter = outs.into_iter();
        let r1 = iter.next().unwrap();
        let r2 = iter.next().unwrap();
        (r1, r2)
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

impl Game<BoardState> for ChessEP {
    fn predict(
        &self,
        node: &ArcRefNode<<BoardState as State>::Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<<BoardState as State>::Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<<BoardState as State>::Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
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
