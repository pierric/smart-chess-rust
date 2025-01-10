use ndarray::{Array1, Ix1};
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::collections::{HashMap, HashSet};

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
    steps: Vec<(chess::Move, Vec<(chess::Move, u32)>)>, apply_mirror: bool
) -> PyResult<Vec<(PyObject, PyObject, PyObject, PyObject)>> {
    let mut history = chess::BoardHistory::new(game::LOOKBACK);
    let mut board_state = chess::BoardState::new();

    let mut ret = Vec::new();

    let flip_color = chess::Color::Black;

    Python::with_gil(|py| {
        for (mov, num_act) in steps.iter() {
            let num_act_dict: HashMap<chess::Move, u32> = num_act.iter().cloned().collect();
            let step = if apply_mirror {
                board_state.to_board().rotate()
            } else {
                board_state.to_board()
            };

            // rotate the move if black
            let rotate_and_encode = |m: &chess::Move| -> i32 {
                if step.turn == flip_color {
                    m.rotate().encode()
                } else {
                    m.encode()
                }
            };

            let legal_moves = board_state.legal_moves();
            // latter boards stay at the front
            history.push_front(&step);

            let encoded_boards = history.view(step.turn == flip_color);
            let encoded_meta = step.encode_meta();

            let moves_indices: Vec<i32> = legal_moves.iter().map(rotate_and_encode).collect();

            // sanity check of the input matching the legal moves
            let set_moves_in: HashSet<chess::Move> = num_act_dict.keys().cloned().collect();
            let set_moves_exp: HashSet<chess::Move> = legal_moves.into_iter().collect();

            let diff: Vec<_> = set_moves_in.symmetric_difference(&set_moves_exp).collect();
            if diff.len() > 0 {
                for x in diff {
                    println!("{x}");
                }
                panic!("inconsistent moves");
            }

            let sum_num_act: u32 = num_act_dict.values().sum();
            let mut encoded_dist = Array1::<f32>::zeros((8 * 8 * 73,));

            for (mov, cnt) in &num_act_dict {
                let ind = rotate_and_encode(mov);
                let val = *cnt as f32 / (sum_num_act as f32 + 1e-5);
                encoded_dist[Ix1(ind as usize)] = val;
            }

            let encoded_boards: Bound<'_, PyArray3<i32>> = PyArray3::from_array(py, &encoded_boards);
            let encoded_meta: Bound<'_, PyArray3<i32>> = PyArray3::from_array(py, &encoded_meta);
            let encoded_dist: Bound<'_, PyArray1<f32>> = PyArray1::from_array(py, &encoded_dist);

            let move_indices = moves_indices.into_pyobject(py).unwrap();

            ret.push((
                encoded_boards.unbind().into_any(),
                encoded_meta.unbind().into_any(),
                encoded_dist.unbind().into_any(),
                move_indices.unbind(),
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
        let encoded_boards: Bound<'_, PyArray3<i32>> = PyArray3::from_array(py, &board.encode_pieces());
        let encoded_meta: Bound<'_, PyArray3<i32>> = PyArray3::from_array(py, &board.encode_meta());
        Ok((encoded_boards.unbind().into_any(), encoded_meta.unbind().into_any()))
    })
}

#[allow(dead_code)]
struct ChessEngineState {
    chess: game::ChessTS,
    board: chess::BoardState,
    root: mcts::ArcRefNode<<chess::BoardState as game::State>::Step>,
    cursor: mcts::Cursor<<chess::BoardState as game::State>::Step>,
}


// NOTE: it is not possible to be Send safely without adding locks, as the cursor part
// carries mutable data. But for the use case, it is just right to fake a Send impl,
// so that we can wrap it into a PyCapsule.
unsafe impl Send for ChessEngineState {}

#[pyfunction]
fn play_new(checkpoint: &str) -> PyResult<PyObject> {
    let device = tch::Device::Cuda(0);
    let chess = game::ChessTS {
        model: tch::CModule::load_on_device(checkpoint, device).unwrap(),
        device: device,
    };

    let board = chess::BoardState::new();
    let (cursor, root) = mcts::Cursor::new(mcts::Node {
        step: (None::<chess::Move>, chess::Color::White),
        depth: 0,
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    });
    let state = ChessEngineState {
        chess: chess,
        board: board,
        root: root,
        cursor: cursor,
    };

    Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, state, None).unwrap();
        Ok(capsule.unbind().into_any())
    })
}

#[pyfunction]
fn play_mcts(state: Py<PyCapsule>, rollout: i32, cpuct: f32) {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<ChessEngineState>() };
        mcts::mcts(
            &state.chess, state.cursor.arc(), &state.board, rollout, Some(cpuct)
        );
    })
}

#[pyfunction]
fn play_inspect(state: Py<PyCapsule>) -> PyResult<(PyObject, PyObject)>{
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<ChessEngineState>() };
        let node = state.cursor.current();
        let q_value = node.q_value;
        let num_act_children: Vec<(chess::Move, i32, f32)> = node
            .children
            .iter()
            .map(|n| {
                let n = n.borrow();
                (n.step.0.unwrap(), n.num_act, n.q_value)
            })
            .collect();

        let q = q_value.into_pyobject(py).unwrap().unbind().into_any();
        let a = num_act_children.into_pyobject(py).unwrap().unbind().into_any();
        Ok((q, a))
    })
}

#[pyfunction]
fn play_dump_search_tree(state: Py<PyCapsule>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<ChessEngineState>() };
        let json = serde_json::to_string(&*state.root.borrow()).unwrap();
        let json_module = py.import("json")?;
        let ret = json_module.getattr("loads")?.call1((json,))?.unbind();
        Ok(ret)
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libsmartchess(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(encode_move, m)?)?;
    m.add_function(wrap_pyfunction!(encode_steps, m)?)?;
    m.add_function(wrap_pyfunction!(play_new, m)?)?;
    m.add_function(wrap_pyfunction!(play_mcts, m)?)?;
    m.add_function(wrap_pyfunction!(play_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(play_dump_search_tree, m)?)?;
    Ok(())
}
