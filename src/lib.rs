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
fn encode_move(turn: chess::Color, mov: chess::Move) -> PyResult<i32> {
    Ok(if turn == chess::Color::Black {
        mov.rotate().encode()
    } else {
        mov.encode()
    })
}

#[pyfunction]
fn encode_steps(
    steps: Vec<(chess::Move, Vec<u32>)>, apply_mirror: bool
) -> PyResult<Vec<(PyObject, PyObject, PyObject, PyObject)>> {
    let mut history = chess::BoardHistory::new(game::LOOKBACK);
    let mut board_state = chess::BoardState::new();

    let mut ret = Vec::new();

    let flip_color = chess::Color::Black;

    Python::with_gil(|py| {
        for (mov, num_act) in steps.iter() {
            let step = if apply_mirror {
                board_state.to_board().rotate()
            } else {
                board_state.to_board()
            };
            let legal_moves = board_state.legal_moves();
            // latter boards stay at the front
            history.push_front(&step);

            let encoded_boards = history.view(step.turn == flip_color);
            let encoded_meta = step.encode_meta();

            // rotate the move if black
            let moves_indices: Vec<i32> = legal_moves
                .iter()
                .map(|m| {
                    if step.turn == flip_color {
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

            let encoded_boards: &PyArray3<i32> = PyArray3::from_array(py, &encoded_boards);
            let encoded_meta: &PyArray3<i32> = PyArray3::from_array(py, &encoded_meta);
            let encoded_dist: &PyArray1<f32> = PyArray1::from_array(py, &encoded_dist);

            ret.push((
                encoded_boards.into_py(py),
                encoded_meta.into_py(py),
                encoded_dist.into_py(py),
                moves_indices.into_py(py),
            ));
            board_state.next(mov);
        }
    });

    Ok(ret)
}

#[pyfunction]
fn encode_board(view: chess::Color, board: chess::Board) -> PyResult<(PyObject, PyObject)> {
    Python::with_gil(|py| {
        let board = if view == chess::Color::White {
            board
        } else {
            board.rotate()
        };
        let encoded_boards: &PyArray3<i32> = PyArray3::from_array(py, &board.encode_pieces());
        let encoded_meta: &PyArray3<i32> = PyArray3::from_array(py, &board.encode_meta());
        Ok((encoded_boards.into_py(py), encoded_meta.into_py(py)))
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libsmartchess(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(encode_move, m)?)?;
    m.add_function(wrap_pyfunction!(encode_steps, m)?)?;
    Ok(())
}
