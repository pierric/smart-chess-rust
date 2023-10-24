use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use crate::chess::{PieceType, Move, EncodeError};

const _PROMOTIONS: [(Option<PieceType>, i32); 3] = [
    (Some(PieceType::Knight), 0),
    (Some(PieceType::BISHOP), 1),
    (Some(PieceType::ROOK), 2),
];

const _TYPE_OFFSET: i32 = 64;

fn encode(mov: Move) -> Result<i32, EncodeError>  {
    let _promotions = HashMap::from(_PROMOTIONS);

    let is_underpromotion =
        _promotions.contains_key(&mov.promotion)
        && mov.from.rank == 6
        && mov.to.rank == 7;

    if !is_underpromotion {
        return Err(EncodeError::NotPromotion);
    }

    let delta_file = mov.to.file - mov.from.file;

    assert!(-1 <= delta_file && delta_file <= 1);

    let direction_idx = delta_file + 1; // 0, 1, 2
    let promotion_idx = _promotions[&mov.promotion];

    Python::with_gil(|py| {
        let np = match py.import("numpy") {
            Ok(np) => np,
            Err(e) => return Err(EncodeError::PythonError(e)),
        };
        let locals = [
            ("np", np.to_object(py)),
            ("direction_idx", direction_idx.to_object(py)),
            ("promotion_idx", promotion_idx.to_object(py)),
            ("_TYPE_OFFSET", _TYPE_OFFSET.to_object(py)),
            ("from_rank", mov.from.rank.to_object(py)),
            ("from_file", mov.from.file.to_object(py)),
        ].into_py_dict(py);

        py.eval(
            r#"underpromotion_type = np.ravel_multi_index((direction_idx, promotion_idx), (3, 3))
               move_type = _TYPE_OFFSET + underpromotion_type;
               np.ravel_multi_index((from_rank, from_file, move_type), (8, 8, 73)"#,
            None,
            Some(locals)
        ).and_then(|r| r.extract::<i32>()).map_err(EncodeError::PythonError)
    })
}
