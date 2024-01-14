use crate::chess::EncodeError;
use crate::chess::Move;
// use pyo3::prelude::*;
// use pyo3::types::IntoPyDict;
use std::collections::HashMap;

const _DIRECTIONS: [((i32, i32), i32); 8] = [
    ((2, 1), 0),
    ((1, 2), 1),
    ((-1, 2), 2),
    ((-2, 1), 3),
    ((-2, -1), 4),
    ((-1, -2), 5),
    ((1, -2), 6),
    ((2, -1), 7),
];

const _TYPE_OFFSET: i32 = 56;

pub fn encode(mov: &Move) -> Result<i32, EncodeError> {
    let delta = (mov.to.rank - mov.from.rank, mov.to.file - mov.from.file);
    let _directions = HashMap::from(_DIRECTIONS);

    if !_directions.contains_key(&delta) {
        return Err(EncodeError::NotKnightMove);
    }

    let knight_move_type = _directions[&delta];
    let move_type = _TYPE_OFFSET + knight_move_type;

    return Ok(mov.from.rank * 8 * 73 + mov.from.file * 73 + move_type);

    // Python::with_gil(|py| {
    //     let np = match py.import("numpy") {
    //         Ok(np) => np,
    //         Err(e) => return Err(EncodeError::PythonError(e)),
    //     };
    //     let locals = [
    //         ("np", np.to_object(py)),
    //         ("from_rank", mov.from.rank.to_object(py)),
    //         ("from_file", mov.from.file.to_object(py)),
    //         ("move_type", move_type.to_object(py)),
    //     ]
    //     .into_py_dict(py);
    //     let code = "np.ravel_multi_index((from_rank, from_file, move_type), (8, 8, 73))";
    //     py.eval(code, None, Some(locals))
    //         .and_then(|r| r.extract::<i32>())
    //         .map_err(EncodeError::PythonError)
    // })
}
