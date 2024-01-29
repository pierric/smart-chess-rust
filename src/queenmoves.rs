use crate::chess::{EncodeError, Move, PieceType};

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

    let direction_idx = match (delta0.signum(), delta1.signum()) {
        (-1, -1) => 5,
        (-1, 0)  => 4,
        (-1, 1)  => 3,
        (0, -1)  => 6,
        (0, 1)   => 2,
        (1, -1)  => 7,
        (1, 0)   => 0,
        (1, 1)   => 1,
        _ => todo!("impossible direction for a queue move")
    };
    let move_type = direction_idx * 7 + distance_idx;
    let action = mov.from.rank * 8 * 73 + mov.from.file * 73 + move_type;
    Ok(action)
}
