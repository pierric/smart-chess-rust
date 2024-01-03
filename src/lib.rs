use ndarray::{Array1, Ix1};
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;

pub mod chess;
pub mod game;
pub mod knightmoves;
pub mod mcts;
pub mod queenmoves;
pub mod underpromotions;

#[pyfunction]
fn encode(steps: Vec<(chess::Move, Vec<u32>)>) -> PyResult<Vec<(PyObject, PyObject, PyObject, PyObject)>> {
    let mut history = chess::BoardHistory::new(game::LOOKBACK);
    let mut board_state = chess::BoardState::new();

    let mut ret = Vec::new();

    Python::with_gil(|py| {
        for (mov, num_act) in steps.iter() {
            let step = board_state.to_board();
            let legal_moves = board_state.legal_moves();
            board_state.next(mov);
            history.push(&step);

            let rotate = step.turn == chess::Color::Black;
            let encoded_boards = history.view(rotate);
            let encoded_meta = step.encode_meta();

            let moves_indices: Vec<i32> = legal_moves
                .iter()
                .map(|m| {
                    if rotate {
                        m.rotate().encode()
                    } else {
                        m.encode()
                    }
                })
                .collect();
            let sum_num_act: u32 = num_act.iter().sum();
            let moves_values: Vec<f32> = num_act
                .iter()
                .map(|v| *v as f32 / (sum_num_act as f32 + 1e-5))
                .collect();

            let mut encoded_dist = Array1::<f32>::zeros((8 * 8 * 73,));
            for (k, v) in moves_indices.iter().zip(moves_values.into_iter()) {
                encoded_dist[Ix1(*k as usize)] = v;
            }

            let encoded_boards: &PyArray3<u32> = PyArray3::from_array(py, &encoded_boards);
            let encoded_meta: &PyArray3<u32> = PyArray3::from_array(py, &encoded_meta);
            let encoded_dist: &PyArray1<f32> = PyArray1::from_array(py, &encoded_dist);

            ret.push((
                encoded_boards.into_py(py),
                encoded_meta.into_py(py),
                encoded_dist.into_py(py),
                moves_indices.into_py(py),
            ));
        }
    });

    Ok(ret)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libencoder(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    Ok(())
}
