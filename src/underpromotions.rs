use crate::chess::{EncodeError, Move, PieceType};
// use pyo3::prelude::*;
// use pyo3::types::IntoPyDict;
use std::collections::HashMap;

const _PROMOTIONS: [(Option<PieceType>, i32); 3] = [
    (Some(PieceType::Knight), 0),
    (Some(PieceType::BISHOP), 1),
    (Some(PieceType::ROOK), 2),
];

const _TYPE_OFFSET: i32 = 64;

pub fn encode(mov: &Move) -> Result<i32, EncodeError> {
    let _promotions = HashMap::from(_PROMOTIONS);

    let is_underpromotion =
        _promotions.contains_key(&mov.promotion) && mov.from.rank == 6 && mov.to.rank == 7;

    if !is_underpromotion {
        return Err(EncodeError::NotPromotion);
    }

    let delta_file = mov.to.file - mov.from.file;

    assert!(-1 <= delta_file && delta_file <= 1);

    let direction_idx = delta_file + 1; // 0, 1, 2
    let promotion_idx = _promotions[&mov.promotion];

    let underpromotion_type = direction_idx * 3 + promotion_idx;
    let move_type = _TYPE_OFFSET + underpromotion_type;
    return Ok(mov.from.rank * 8 * 73 + mov.from.file * 73 + move_type);

    //     Python::with_gil(|py| {
    //         let np = match py.import("numpy") {
    //             Ok(np) => np,
    //             Err(e) => return Err(EncodeError::PythonError(e)),
    //         };
    //         let locals = [
    //             ("np", np.to_object(py)),
    //             ("direction_idx", direction_idx.to_object(py)),
    //             ("promotion_idx", promotion_idx.to_object(py)),
    //             ("_TYPE_OFFSET", _TYPE_OFFSET.to_object(py)),
    //             ("from_rank", mov.from.rank.to_object(py)),
    //             ("from_file", mov.from.file.to_object(py)),
    //         ]
    //         .into_py_dict(py);

    //         let code = r#"
    // underpromotion_type = np.ravel_multi_index((direction_idx, promotion_idx), (3, 3))
    // move_type = _TYPE_OFFSET + underpromotion_type;
    // ret = np.ravel_multi_index((from_rank, from_file, move_type), (8, 8, 73))"#;

    //         py.run(code, None, Some(locals)).unwrap();
    //         locals
    //             .get_item("ret")
    //             .unwrap()
    //             .unwrap()
    //             .extract()
    //             .map_err(EncodeError::PythonError)
    //     })
}
