use ndarray::{Array1, Ix1};
use numpy::array::{PyArray1, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::collections::{HashMap, HashSet};
use std::cell::{RefCell, RefMut};
use std::sync::Arc;

pub mod chess;
pub mod game;
pub mod knightmoves;
pub mod mcts;
pub mod queenmoves;
pub mod underpromotions;
pub mod backends;

#[allow(non_camel_case_types)]
mod jina {
    tonic::include_proto!("jina");
}
mod docarray {
    tonic::include_proto!("docarray");
}

#[pyfunction]
fn chess_encode_move(turn: chess::Color, mov: chess::Move) -> PyResult<i32> {
    Ok(if turn == chess::Color::Black {
        mov.rotate().encode()
    } else {
        mov.encode()
    })
}

#[pyfunction]
fn chess_encode_steps(
    steps: Vec<(chess::Move, Vec<(chess::Move, u32)>)>, apply_mirror: bool
) -> PyResult<Vec<(PyObject, PyObject, PyObject, PyObject)>> {
    let mut history = chess::BoardHistory::new(chess::LOOKBACK);
    let mut board_state = chess::BoardState::new();

    let mut ret = Vec::new();

    let flip_color = chess::Color::Black;

    Python::with_gil(|py| {
        for (next_mov, num_act) in steps.iter() {
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

            if !set_moves_exp.contains(next_mov) {
                panic!("num_act table doesn't include the next move");
            }

            let sum_num_act: u32 = num_act_dict.values().sum();
            let mut encoded_dist = Array1::<f32>::zeros((8 * 8 * 73,));

            for (mov, cnt) in &num_act_dict {
                let ind = rotate_and_encode(mov);
                let val = *cnt as f32 / (sum_num_act as f32 + 1e-5);
                encoded_dist[Ix1(ind as usize)] = val;
            }

            let encoded_boards: Bound<'_, PyArray3<i8>> = PyArray3::from_array(py, &encoded_boards);
            let encoded_meta: Bound<'_, PyArray1<i32>> = PyArray1::from_array(py, &encoded_meta);
            let encoded_dist: Bound<'_, PyArray1<f32>> = PyArray1::from_array(py, &encoded_dist);

            let move_indices = moves_indices.into_pyobject(py).unwrap();

            ret.push((
                encoded_boards.unbind().into_any(),
                encoded_meta.unbind().into_any(),
                encoded_dist.unbind().into_any(),
                move_indices.unbind(),
            ));
            board_state.next(next_mov);
        }
    });

    Ok(ret)
}

#[pyfunction]
fn chess_encode_board(view: chess::Color, board: chess::Board) -> PyResult<(PyObject, PyObject)> {
    Python::with_gil(|py| {
        let board = if view == chess::Color::White {
            board
        } else {
            board.rotate()
        };
        let encoded_boards: Bound<'_, PyArray3<i8>> = PyArray3::from_array(py, &board.encode_pieces());
        let encoded_meta: Bound<'_, PyArray1<i32>> = PyArray1::from_array(py, &board.encode_meta());
        Ok((encoded_boards.unbind().into_any(), encoded_meta.unbind().into_any()))
    })
}

#[allow(dead_code)]
struct ChessEngineState {
    chess: backends::torch::ChessEP,
    board: chess::BoardState,
    root: mcts::ArcRefNode<<chess::BoardState as game::State>::Step>,
    cursor: mcts::Cursor<<chess::BoardState as game::State>::Step>,
}


// NOTE: it is not possible to be Send safely without adding locks, as the cursor part
// carries mutable data. But for the use case, it is just right to fake a Send impl,
// so that we can wrap it into a PyCapsule.
unsafe impl Send for ChessEngineState {}

#[pyfunction]
fn chess_play_new(checkpoint: &str, device: &str, initial_moves: Vec<chess::Move>) -> PyResult<PyObject> {
    let device = match device {
        "cpu" => tch::Device::Cpu,
        "cuda" => tch::Device::Cuda(0),
        "mps" => tch::Device::Mps,
        _ => todo!("Unsupported device name"),
    };

    //let chess = chess::ChessTS {
    //    model: tch::CModule::load_on_device(checkpoint, device).unwrap(),
    //    device: device,
    //};
    let chess = backends::torch::ChessEP {
        model: aotinductor::ModelPackage::new(checkpoint).unwrap(),
        device: device,
    };

    let mut board = chess::BoardState::new();
    let mut turn = chess::Color::White;
    let (mut cursor, root) = mcts::Cursor::new(mcts::Node {
        step: chess::Step(None::<chess::Move>, chess::Color::White),
        depth: 0,
        q_value: 0.,
        num_act: 0,
        parent: None,
        children: Vec::new(),
    });

    for mov in initial_moves {
        board.next(&mov);
        let child = {
            Arc::new(RefCell::new(mcts::Node {
                step: chess::Step(Some(mov), !turn),
                depth: cursor.current().depth + 1,
                q_value: 0.,
                num_act: 0,
                parent: Some(cursor.as_weak()),
                children: Vec::new(),
            }))
        };
        cursor.current_mut().children = vec![child];
        cursor.navigate_down(0);
        turn = !turn;
    }

    let state = RefCell::new(ChessEngineState {
        chess: chess,
        board: board,
        root: root,
        cursor: cursor,
    });

    Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, state, None).unwrap();
        Ok(capsule.unbind().into_any())
    })
}

#[pyfunction]
fn chess_play_mcts(state: Py<PyCapsule>, rollout: i32, cpuct: f32) {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<RefCell<ChessEngineState>>().borrow() };
        mcts::mcts(
            &state.chess, state.cursor.arc(), &state.board, rollout, Some(cpuct), true
        );
    })
}

#[pyfunction]
fn chess_play_step(state: Py<PyCapsule>, temp: f32) {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<RefCell<ChessEngineState>>().borrow_mut() };
        let (mut cursor, mut board) = RefMut::map_split(state, |o| (&mut o.cursor, &mut o.board));
        mcts::step(&mut *cursor, &mut *board, temp);
    })
}

#[pyfunction]
fn chess_play_inspect(state: Py<PyCapsule>) -> PyResult<(PyObject, PyObject, PyObject, PyObject)>{
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<RefCell<ChessEngineState>>().borrow() };
        let board = Py::clone_ref(&state.board.python_object, py);
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

        let mut move_stack: Vec<chess::Move> = Vec::new();
        let mut pointer = state.cursor.clone();
        loop {
            let mov = pointer.current().step.0;
            if mov.is_none() {
                break;
            }
            move_stack.push(mov.unwrap());
            pointer.navigate_up();
        }

        let m = move_stack.into_pyobject(py).unwrap().unbind().into_any();
        let q = q_value.into_pyobject(py).unwrap().unbind().into_any();
        let a = num_act_children.into_pyobject(py).unwrap().unbind().into_any();
        Ok((board, m, q, a))
    })
}

#[pyfunction]
fn chess_play_dump_search_tree(state: Py<PyCapsule>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<RefCell<ChessEngineState>>().borrow() };
        let json = serde_json::to_string(&*state.root.borrow()).unwrap();
        let json_module = py.import("json")?;
        let ret = json_module.getattr("loads")?.call1((json,))?.unbind();
        Ok(ret)
    })
}

#[pyfunction]
fn chess_play_inference(state: Py<PyCapsule>, full_distr: bool) -> PyResult<(PyObject, PyObject, PyObject)> {
    Python::with_gil(|py| {
        let state = unsafe { state.bind(py).reference::<RefCell<ChessEngineState>>().borrow() };

        let (steps, prior, outcome) = backends::torch::chess_tch_predict(&state.chess, &state.cursor.arc(), &state.board, false, full_distr);
        let steps = steps.into_pyobject(py)?.unbind().into_any();
        let prior = prior.into_pyobject(py)?.unbind().into_any();
        let outcome = outcome.into_pyobject(py)?.unbind().into_any();
        Ok((steps, prior, outcome))
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libsmartchess(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chess_encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(chess_encode_move, m)?)?;
    m.add_function(wrap_pyfunction!(chess_encode_steps, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_new, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_mcts, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_step, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_dump_search_tree, m)?)?;
    m.add_function(wrap_pyfunction!(chess_play_inference, m)?)?;
    Ok(())
}
