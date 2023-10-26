use std::fmt;
use std::ops::Not;
use std::collections::VecDeque;
use pyo3::prelude::{PyErr, Py, PyAny, FromPyObject, Python};
use pyo3::conversion::IntoPy;
use pyo3::intern;
use ndarray::{Array3, Ix3, s};
use crate::{knightmoves, queenmoves, underpromotions};

#[derive(Debug)]
pub enum EncodeError {
    PythonError(PyErr),
    NotKnightMove,
    NotQueenMove,
    NotPromotion,
}

pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
}

#[derive(Copy, Clone)]
pub struct Square {
    pub rank: i32,
    pub file: i32,
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone)]
pub enum PieceType {
    Pawn = 1,
    Knight,
    BISHOP,
    ROOK,
    Queen,
    King,
}


#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone)]
pub enum Color {
    Black = 0,
    White,
}

#[derive(Copy, Clone, FromPyObject)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

pub struct Board<'a> {
    py_object: &'a PyAny,
    pub turn: Color,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
}

pub struct BoardHistoryState {
    piece_map: Vec<(Square, Piece)>,
    repetition2: bool,
    repetition3: bool,
}

pub struct BoardHistory {
    pub size: usize,
    pub history: VecDeque<BoardHistoryState>,
}

impl From<i32> for PieceType {
    fn from(i: i32) -> Self {
        match i {
            1 => PieceType::Pawn,
            2 => PieceType::Knight,
            3 => PieceType::BISHOP,
            4 => PieceType::ROOK,
            5 => PieceType::Queen,
            6 => PieceType::King,
            _ => panic!("Invalid piece type"),
        }
    }
}

impl Not for Color {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl From<i32> for Color {
    fn from(i: i32) -> Self {
        match i {
            0 => Color::Black,
            1 => Color::White,
            _ => panic!("Invalid color"),
        }
    }
}

impl<'a> FromPyObject<'a> for PieceType {
    fn extract(obj: &PyAny) -> Result<Self, PyErr> {
        return obj.extract::<i32>().map(|v| v.into());
    }
}

impl<'a> FromPyObject<'a> for Color {
    fn extract(obj: &PyAny) -> Result<Self, PyErr> {
        return obj.extract::<i32>().map(|v| v.into());
    }
}

impl IntoPy<Py<PyAny>> for Color {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        let v = self == Color::White;
        return v.into_py(py);
    }
}

impl<'a> FromPyObject<'a> for Move {
    fn extract(obj: &PyAny) -> Result<Self, PyErr> {
        let py = obj.py();
        let from = obj.getattr(intern!(py, "from_square")).and_then(|o| o.extract())?;
        let to = obj.getattr(intern!(py, "to_square")).and_then(|o| o.extract())?;
        let promotion = obj.getattr(intern!(py, "promotion")).and_then(|o| o.extract())?;
        return Ok(Move {from, to, promotion});
    }
}

impl<'a> FromPyObject<'a> for Square {
    fn extract(obj: &PyAny) -> Result<Self, PyErr> {
        let square: i32 = obj.extract()?;
        let file = square & 7;
        let rank = square >> 3;
        Ok(Square {rank, file})
    }
}

impl<'a> FromPyObject<'a> for Board<'a>{
    fn extract(obj: &'a PyAny) -> Result<Self, PyErr> {
        let py = obj.py();
        let turn = obj.getattr(intern!(py, "turn"))?.extract()?;
        let halfmove_clock = obj.getattr(intern!(py, "halfmove_clock"))?.extract()?;
        let fullmove_number = obj.getattr(intern!(py, "fullmove_number"))?.extract()?;
        return Ok(Board {
            py_object: obj,
            turn,
            halfmove_clock,
            fullmove_number,
        })
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank, self.file)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.promotion {
            Some(p) => write!(f, "Move ({}, {}, {})", self.from, self.to, p),
            None => write!(f, "Move ({}, {})", self.from, self.to),
        }
    }
}

impl Square {
    pub fn rotate(&self) -> Self {
        return Self {rank: 7 - self.rank, file: 7 - self.file}
    }
}

impl Move {
    pub fn rotate(&self) -> Self {
        let from = self.from.rotate();
        let to = self.to.rotate();
        return Self {from, to, promotion: self.promotion}
    }

    pub fn encode(&self) -> i32 {
        queenmoves::encode(self)
            .or(knightmoves::encode(self))
            .or(underpromotions::encode(self))
            .or(Err(format!("Invalid action {}", self)))
            .unwrap()
    }
}

impl<'a> Board<'a> {
    pub fn piece_map(&self) -> Vec<(Square, Piece)> {
        return self.py_object.call_method0(
            intern!(self.py_object.py(), "square_piece")
        ).unwrap().extract().unwrap()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        return self.py_object.getattr(
            intern!(self.py_object.py(), "legal_moves")
        ).unwrap().extract().unwrap()
    }

    pub fn is_repetition(&self, count: i32) -> bool {
        return self.py_object.call_method1(
            intern!(self.py_object.py(), "is_repetition"), (count,)
         ).unwrap().extract().unwrap()
    }

    pub fn has_kingside_castling_rights(&self, turn: Color) -> bool {
        return self.py_object.call_method1(
            intern!(self.py_object.py(), "has_kingside_castling_rights"),
            (turn,)
        ).unwrap().extract().unwrap()
    }

    pub fn has_queenside_castling_rights(&self, turn: Color) -> bool {
        return self.py_object.call_method1(
            intern!(self.py_object.py(), "has_queenside_castling_rights"),
            (turn,)
        ).unwrap().extract().unwrap()
    }

    fn _encode_board_state(state: &BoardHistoryState) -> Array3<u32> {
        let mut array = Array3::<u32>::zeros((8, 8, 14));

        for (square, piece) in state.piece_map.iter() {
            let piece_type = piece.piece_type;
            let color = piece.color;

            // The first six planes encode the pieces of the active player,
            // the following six those of the active player's opponent. Since
            // this class always stores boards oriented towards the white player,
            // White is considered to be the active player here.
            let offset = if color == Color::White {0} else {6};

            // Chess enumerates piece types beginning with one, which we have
            // to account for
            let idx = piece_type as i32 - 1;

            let rank = square.rank.try_into().unwrap();
            let file = square.file.try_into().unwrap();

            array[Ix3(rank, file, (idx + offset).try_into().unwrap())] = 1;
        }
        // Repetition counters
        array.slice_mut(s![.., .., 12]).fill(state.repetition2 as u32);
        array.slice_mut(s![.., .., 13]).fill(state.repetition3 as u32);

        return array
    }

    pub fn get_state(&self) -> BoardHistoryState {
        let piece_map = self.piece_map();
        let repetition2 = self.is_repetition(2);
        let repetition3 = self.is_repetition(3);
        return BoardHistoryState {piece_map, repetition2, repetition3}
    }

    pub fn encode_pieces(&self) -> Array3<u32> {
        let state = self.get_state();
        return Board::_encode_board_state(&state);
    }

    pub fn encode_meta(&self) -> Array3<u32> {
        let mut meta = Array3::<u32>::zeros((8, 8, 7));
        meta.slice_mut(s![.., .., 0]).fill(self.turn as u32);
        meta.slice_mut(s![.., .., 1]).fill(self.fullmove_number);
        meta.slice_mut(s![.., .., 2]).fill(self.has_kingside_castling_rights(self.turn) as u32);
        meta.slice_mut(s![.., .., 3]).fill(self.has_queenside_castling_rights(self.turn) as u32);
        meta.slice_mut(s![.., .., 4]).fill(self.has_kingside_castling_rights(!self.turn) as u32);
        meta.slice_mut(s![.., .., 5]).fill(self.has_queenside_castling_rights(!self.turn) as u32);
        meta.slice_mut(s![.., .., 6]).fill(self.halfmove_clock);
        return meta
    }
}

impl BoardHistoryState {
    fn rotate(&self) -> Self {
        let piece_map = self.piece_map.iter().map(|(square, piece)| {
            let square = square.rotate();
            let piece = Piece {piece_type: piece.piece_type, color: !piece.color};
            return (square, piece);
        }).collect();
        return BoardHistoryState {
            piece_map,
            repetition2: self.repetition2,
            repetition3: self.repetition3
        };
    }
}

impl BoardHistory {
    pub fn new(size: usize) -> Self {
        return BoardHistory {size: size, history: VecDeque::with_capacity(size)};
    }

    pub fn push(&mut self, board: Board) {
        self.history.pop_back();
        self.history.push_front(board.get_state());
    }

    pub fn view(&self, rotate: bool) -> Array3<u32> {
        let mut full = Array3::<u32>::zeros((8, 8, 14 * self.size));

        for idx in 0..self.history.len() {
            let state = self.history.get(idx).unwrap();
            let array = if rotate {
                Board::_encode_board_state(&state.rotate())
            } else {
                Board::_encode_board_state(state)
            };
            full.slice_mut(s![.., .., 14 * idx .. 14 * (idx + 1)]).assign(&array);
        }

        return full
    }
}
