use numpy::{PyArray, PyArrayMethods};
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::*;
use ndarray::{Array1, Array3};
use c_str_macro::c_str;

use crate::chess::{Color, Move, Step, BoardState, _encode};
use crate::mcts::ArcRefNode;
use crate::game;

pub struct ChessPy {
    pub model: Py<PyAny>,
    pub device: String,
}

impl game::Game<BoardState> for ChessPy {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
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
            &self.model,
            &self.device,
            encoded_boards,
            encoded_meta,
            turn,
            &legal_moves,
        );

        let moves_distr = game::post_process_distr(moves_distr, argmax);
        let next_steps = legal_moves
            .into_iter()
            .map(|m| Step(Some(m), !turn))
            .collect();
        return (next_steps, moves_distr, score);
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

fn call_py_model(
    model: &Py<PyAny>,
    device: &str,
    boards: Array3<i8>,
    meta: Array1<i32>,
    turn: Color,
    steps: &Vec<Move>,
) -> (Vec<f32>, f32) {
    Python::with_gil(|py| {
        let encoded_boards = PyArray3::from_array(py, &boards);
        let encoded_meta = PyArray1::from_array(py, &meta);

        let code = c_str!(r#"
encoded_meta = encode_meta[None,:].repeat(64).reshape((8, 8, 7))
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

