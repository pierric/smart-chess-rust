use crate::chess::{EncodeError, Move, PieceType};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

pub fn encode(mov: &Move) -> Result<i32, EncodeError> {
    let is_queen_move_promotion =
        mov.promotion.is_none() | (mov.promotion == Some(PieceType::Queen));
    let delta0 = mov.to.rank - mov.from.rank;
    let delta1 = mov.to.file - mov.from.file;

    let is_horizontal = delta0 == 0;
    let is_vertical = delta1 == 0;
    let is_diagonal = delta0.abs() == delta1.abs();
    let is_queen_move = (is_horizontal || is_vertical || is_diagonal) && is_queen_move_promotion;
    let distance = delta0.abs().max(delta1.abs());
    let distance_idx = distance - 1;

    if !is_queen_move {
        return Err(EncodeError::NotQueenMove);
    }

    Python::with_gil(|py| {
        let np = match py.import("numpy") {
            Err(e) => return Err(EncodeError::PythonError(e)),
            Ok(np) => np,
        };
        let locals = [
            ("np", np.to_object(py)),
            ("delta0", delta0.to_object(py)),
            ("delta1", delta1.to_object(py)),
            ("from_rank", mov.from.rank.to_object(py)),
            ("from_file", mov.from.file.to_object(py)),
            ("distance_idx", distance_idx.to_object(py)),
        ]
        .into_py_dict(py);

        let code = r#"
direction = np.sign([delta0, delta1])
direction_table = [
    [4,4,3],
    [5,8,2],
    [6,0,1],
]
direction_idx = np.array(direction_table, np.int32)[tuple(direction+1)]
move_type = np.ravel_multi_index(
    ([direction_idx, distance_idx]),
    (8,7)
)

action = np.ravel_multi_index(
    ((from_rank, from_file, move_type)),
    (8, 8, 73)
)"#;
        py.run(code, None, Some(&locals)).unwrap();
        locals
            .get_item("action")
            .unwrap()
            .unwrap()
            .extract()
            .map_err(EncodeError::PythonError)
    })
}
