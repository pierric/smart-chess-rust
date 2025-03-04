use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Not;
use std::cmp;
use cached::{SizedCache, Cached};
use ndarray::{s, Array3, Ix3};
use once_cell::sync::Lazy;
use pyo3::conversion::IntoPyObject;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use numpy::{PyArray};
use numpy::array::{PyArray1, PyArray3};
use numpy::PyArrayMethods;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::Tensor;
use short_uuid::ShortUuid;
use c_str_macro::c_str;

use crate::{game, mcts, knightmoves, queenmoves, underpromotions};
use game::{Game, TchModel};
use mcts::{Cursor, ArcRefNode};

pub const LOOKBACK: usize = 8;

static CHESS_MODULE: Lazy<Py<PyModule>> =
    Lazy::new(|| Python::with_gil(|py| py.import("chess").unwrap().into()));

#[derive(Debug)]
pub enum EncodeError {
    NotKnightMove,
    NotQueenMove,
    NotPromotion,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub drop: Option<PieceType>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Square {
    pub rank: i32,
    pub file: i32,
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum PieceType {
    Pawn = 1,
    Knight,
    BISHOP,
    ROOK,
    Queen,
    King,
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Color {
    Black = 0,
    White,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Step(pub Option<Move>, pub Color);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, FromPyObject)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Board {
    pub last_move: Option<Move>,
    pub turn: Color,
    pub piece_map: Vec<(Square, Piece)>,
    pub repetition2: bool,
    pub repetition3: bool,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub has_kingside_castling_rights: (bool, bool),
    pub has_queenside_castling_rights: (bool, bool),
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Termination {
    Checkmate = 1,
    Stalemate,
    InsufficientMaterial,
    SeventyfiveMoves,
    FivefoldRepetition,
    FiftyMoves,
    ThreefoldRepetition,
    VariantWin,
    VariantLoss,
    VariantDraw,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub termination: Termination,
    pub winner: Option<Color>,
}

pub struct BoardState {
    pub python_object: Py<PyAny>,
}

pub struct BoardHistory {
    pub size: usize,
    pub history: VecDeque<Board>,
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
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        return obj.extract::<i32>().map(|v| v.into());
    }
}

impl<'a> IntoPyObject<'a> for PieceType {
    type Target = PyInt;
    type Output = Bound<'a, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        (self as i32).into_pyobject(py)
    }
}

impl<'a> FromPyObject<'a> for Color {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        return obj.extract::<i32>().map(|v| v.into());
    }
}

impl<'a> IntoPyObject<'a> for Color {
    type Target = PyBool;
    type Output = Borrowed<'a, 'a, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        let v: bool = self == Color::White;
        return v.into_pyobject(py);
    }
}

impl<'a> FromPyObject<'a> for Move {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        let py = obj.py();
        let from = obj
            .getattr(intern!(py, "from_square"))
            .and_then(|o| o.extract())?;
        let to = obj
            .getattr(intern!(py, "to_square"))
            .and_then(|o| o.extract())?;
        let promotion = obj
            .getattr(intern!(py, "promotion"))
            .and_then(|o| o.extract())?;
        let drop = obj.getattr(intern!(py, "drop")).and_then(|o| o.extract())?;
        return Ok(Move {
            from,
            to,
            promotion,
            drop,
        });
    }
}

impl<'a> IntoPyObject<'a> for Move {
    type Target = PyAny;
    type Output = Bound<'a, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        let from = self.from.into_pyobject(py).unwrap().into_any().unbind();
        let to = self.to.into_pyobject(py).unwrap().into_any().unbind();
        let promotion = match self.promotion {
            None => py.None(),
            Some(pt) => pt.into_pyobject(py).unwrap().into_any().unbind(),
        };
        let drop = match self.drop {
            None => py.None(),
            Some(pt) => pt.into_pyobject(py).unwrap().into_any().unbind(),
        };
        let ctor = CHESS_MODULE.bind(py).getattr("Move").unwrap();
        Ok(ctor.call1((from, to, promotion, drop)).unwrap())
    }
}

impl Serialize for Move {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.uci())
    }
}

impl<'a> Deserialize<'a> for Move {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        struct MoveVisitor;

        impl<'a> serde::de::Visitor<'a> for MoveVisitor {
            type Value = Move;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a move")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Move::from_uci(v))
            }
        }

        deserializer.deserialize_string(MoveVisitor)
    }
}

impl Serialize for Step {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (self.0, self.1).serialize(serializer)
    }
}

impl<'a> IntoPyObject<'a> for Step {
    type Target = PyTuple;
    type Output = Bound<'a, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        (self.0, self.1).into_pyobject(py)
    }
}

impl<'a> FromPyObject<'a> for Square {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        let square: i32 = obj.extract()?;
        let file = square & 7;
        let rank = square >> 3;
        Ok(Square { rank, file })
    }
}

impl<'a> IntoPyObject<'a> for Square {
    type Target = PyAny;
    type Output = Bound<'a, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        CHESS_MODULE
            .bind(py)
            .getattr("square")
            .and_then(|square| square.call1((self.file, self.rank)))
            .unwrap()
            .into_pyobject(py)
    }
}

impl From<i32> for Termination {
    fn from(i: i32) -> Self {
        match i {
            1 => Termination::Checkmate,
            2 => Termination::Stalemate,
            3 => Termination::InsufficientMaterial,
            4 => Termination::SeventyfiveMoves,
            5 => Termination::FivefoldRepetition,
            6 => Termination::FiftyMoves,
            7 => Termination::ThreefoldRepetition,
            8 => Termination::VariantWin,
            9 => Termination::VariantLoss,
            10 => Termination::VariantDraw,
            _ => panic!("Invalid Termination"),
        }
    }
}

impl<'a> FromPyObject<'a> for Termination {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        return obj
            .getattr(intern!(obj.py(), "value"))?
            .extract::<i32>()
            .map(|v| v.into());
    }
}

impl<'a> FromPyObject<'a> for Outcome {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        let py = obj.py();
        let term = obj.getattr(intern!(py, "termination"))?.extract()?;
        let winner = obj.getattr(intern!(py, "winner"))?.extract()?;
        return Ok(Outcome {
            termination: term,
            winner: winner,
        });
    }
}

impl<'a> FromPyObject<'a> for Board {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        let py = obj.py();

        let turn: i32 = obj.getattr(intern!(py, "turn"))?.extract()?;
        let turn: Color = turn.into();
        let halfmove_clock = obj.getattr(intern!(py, "halfmove_clock"))?.extract()?;
        let fullmove_number = obj.getattr(intern!(py, "fullmove_number"))?.extract()?;
        let piece_map: Bound<'a, PyDict> = obj
            .call_method0(intern!(py, "piece_map"))
            ?.downcast_into()
            .map_err(PyErr::from)?;
        let piece_map = piece_map
            .iter()
            .map(|(square, piece)| (square.extract().unwrap(), piece.extract().unwrap()))
            .collect();
        let repetition2 = obj
            .call_method1(intern!(py, "is_repetition"), (2,))?
            .extract()?;
        let repetition3 = obj
            .call_method1(intern!(py, "is_repetition"), (3,))?
            .extract()?;
        let kingside_castling_1 = obj
            .call_method1(intern!(py, "has_kingside_castling_rights"), (turn,))?
            .extract()?;
        let kingside_castling_2 = obj
            .call_method1(intern!(py, "has_kingside_castling_rights"), (!turn,))?
            .extract()?;
        let queenside_castling_1 = obj
            .call_method1(intern!(py, "has_queenside_castling_rights"), (turn,))?
            .extract()?;
        let queenside_castling_2 = obj
            .call_method1(intern!(py, "has_queenside_castling_rights"), (!turn,))?
            .extract()?;

        let last_move = if fullmove_number == 1 && turn == Color::White {
            None
        } else {
            let mov = obj.call_method0(intern!(py, "peek")).unwrap().extract()?;
            Some(mov)
        };

        return Ok(Board {
            last_move,
            turn,
            piece_map,
            halfmove_clock,
            fullmove_number,
            repetition2,
            repetition3,
            has_kingside_castling_rights: (kingside_castling_1, kingside_castling_2),
            has_queenside_castling_rights: (queenside_castling_1, queenside_castling_2),
        });
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.rank)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move[{}]", self.uci())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move[{}]", self.uci())
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::Black => write!(f, "B"),
            Color::White => write!(f, "W"),
        }
    }
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => write!(f, "#"),
            Some(m) => write!(f, "{}", m),
        }?;
        write!(f, "/{}", self.1)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Board: turn {}", self.turn)?;
        match self.last_move {
            None => Ok(()),
            Some(m) => {
                write!(f, " last move {}", m)
            }
        }
    }
}

impl fmt::Display for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fen())
    }
}

impl PieceType {
    pub fn symbol(&self) -> &'static str {
        match self {
            PieceType::Pawn => "p",
            PieceType::Knight => "n",
            PieceType::BISHOP => "b",
            PieceType::ROOK => "r",
            PieceType::Queen => "q",
            PieceType::King => "k",
        }
    }
}

const SQUARE_NAMES: [[&str; 8]; 8] = [
    ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"],
    ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"],
    ["a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3"],
    ["a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4"],
    ["a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5"],
    ["a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"],
    ["a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"],
    ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"],
];

impl Square {
    pub fn symbol(&self) -> &str {
        SQUARE_NAMES[self.rank as usize][self.file as usize]
    }

    pub fn rotate(&self) -> Self {
        return Self {
            rank: 7 - self.rank,
            file: self.file,  // flipping the board really
        };
    }
}

impl Move {
    pub fn uci(&self) -> String {
        match (self.drop, self.promotion) {
            (Some(p), _) => format!("{}@{}", p.symbol(), self.to.symbol()),
            (None, Some(p)) => format!("{}{}{}", self.from.symbol(), self.to.symbol(), p.symbol()),
            (None, None) => format!("{}{}", self.from.symbol(), self.to.symbol()),
        }
    }

    pub fn from_uci(uci: &str) -> Self {
        Python::with_gil(|py| {
            CHESS_MODULE
                .bind(py)
                .getattr("Move")
                .and_then(|cls| cls.getattr("from_uci"))
                .and_then(|func| func.call1((uci,)))
                .and_then(|mov| mov.extract())
                .unwrap()
        })
    }

    pub fn rotate(&self) -> Self {
        let from = self.from.rotate();
        let to = self.to.rotate();
        return Self {
            from,
            to,
            promotion: self.promotion,
            drop: self.drop,
        };
    }

    pub fn encode(&self) -> i32 {
        queenmoves::encode(self)
            .or(knightmoves::encode(self))
            .or(underpromotions::encode(self))
            .or(Err(format!("Invalid action {}", self)))
            .unwrap()
    }
}

static mut GET_BOARD_CACHE: Lazy<SizedCache<u64, Board>> =
    Lazy::new(|| SizedCache::with_size(50000));

#[allow(static_mut_refs)]
pub fn get_board_from_moves(moves: &Vec<Move>) -> Board {
    let mut hasher = DefaultHasher::new();
    moves.hash(&mut hasher);
    let len_start = cmp::max(moves.len(), 8) - 8;
    let full_key = hasher.finish();

    unsafe {
        match GET_BOARD_CACHE.cache_get(&full_key) {
            Some(v) => v.clone(),
            None => {
                let mut vec = Vec::with_capacity(moves.len());
                let mut state = BoardState::new();
                let mut len = 0;

                for m in moves.iter() {
                    state.next(m);
                    vec.push(m);

                    len += 1;
                    if len >= len_start {

                        let mut hasher = DefaultHasher::new();
                        vec.hash(&mut hasher);
                        let mkey = hasher.finish();

                        GET_BOARD_CACHE.cache_get_or_set_with(
                            mkey, || state.to_board()
                        );
                    }
                }

                GET_BOARD_CACHE.cache_get_or_set_with(
                    full_key, || state.to_board()
                ).clone()
            }
        }
    }
}

impl Board {
    pub fn rotate(&self) -> Self {
        let piece_map = self
            .piece_map
            .iter()
            .map(|(square, piece)| {
                let square = square.rotate();
                let piece = Piece {
                    piece_type: piece.piece_type,
                    color: !piece.color,
                };
                return (square, piece);
            })
            .collect();

        let swap = |(a, b)| (b, a);

        return Self {
            last_move: self.last_move.map(|m| m.rotate()),
            turn: !self.turn,
            piece_map,
            repetition2: self.repetition2,
            repetition3: self.repetition3,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            has_kingside_castling_rights: swap(self.has_kingside_castling_rights),
            has_queenside_castling_rights: swap(self.has_queenside_castling_rights),
        };
    }

    pub fn encode_pieces(&self) -> Array3<i8> {
        let mut array = Array3::<i8>::zeros((8, 8, 14));

        for (square, piece) in self.piece_map.iter() {
            let piece_type = piece.piece_type;
            let color = piece.color;

            // The first six planes encode the pieces of the active player,
            // the following six those of the active player's opponent. Since
            // this class always stores boards oriented towards the white player,
            // White is considered to be the active player here.
            let offset = if color == Color::White { 0 } else { 6 };

            // Chess enumerates piece types beginning with one, which we have
            // to account for
            let idx = piece_type as i32 - 1;

            let rank = square.rank.try_into().unwrap();
            let file = square.file.try_into().unwrap();

            array[Ix3(rank, file, (idx + offset).try_into().unwrap())] = 1;
        }
        // Repetition counters
        array
            .slice_mut(s![.., .., 12])
            .fill(self.repetition2 as i8);
        array
            .slice_mut(s![.., .., 13])
            .fill(self.repetition3 as i8);

        return array;
    }

    pub fn encode_meta(&self) -> Array3<i32> {
        let mut meta = Array3::<i32>::zeros((8, 8, 7));
        meta.slice_mut(s![.., .., 0]).fill(self.turn as i32);
        meta.slice_mut(s![.., .., 1])
            .fill(self.fullmove_number as i32);
        meta.slice_mut(s![.., .., 2])
            .fill(self.has_kingside_castling_rights.0 as i32);
        meta.slice_mut(s![.., .., 3])
            .fill(self.has_queenside_castling_rights.0 as i32);
        meta.slice_mut(s![.., .., 4])
            .fill(self.has_kingside_castling_rights.1 as i32);
        meta.slice_mut(s![.., .., 5])
            .fill(self.has_queenside_castling_rights.1 as i32);
        meta.slice_mut(s![.., .., 6])
            .fill(self.halfmove_clock as i32);
        return meta;
    }
}

impl BoardState {
    pub fn new() -> Self {
        Python::with_gil(|py| {
            let board = CHESS_MODULE
                .bind(py)
                .getattr("Board")
                .and_then(|board| board.call0())
                .unwrap();
            return Self {
                python_object: PyObject::from(board),
            };
        })
    }

    #[allow(dead_code)]
    pub fn turn(&self) -> Color {
        Python::with_gil(|py| {
            self.python_object
                .getattr(py, intern!(py, "turn"))
                .unwrap()
                .extract::<i32>(py)
                .unwrap()
                .into()
        })
    }

    pub fn to_board(&self) -> Board {
        Python::with_gil(|py| self.python_object.extract(py).unwrap())
    }

    #[allow(dead_code)]
    pub fn fen(&self) -> String {
        Python::with_gil(|py| {
            let res = self
                .python_object
                .call_method0(py, intern!(py, "fen"))
                .unwrap();
            res.extract(py).unwrap()
        })
    }

    pub fn outcome(&self) -> Option<Outcome> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("claim_draw", true)?;
            let res =
                self.python_object
                    .call_method(py, intern!(py, "outcome"), (), Some(&kwargs))?;
            res.extract(py)
        })
        .unwrap()
    }

    #[allow(dead_code)]
    pub fn prev(&mut self) -> Option<Move> {
        Python::with_gil(|py| {
            self.python_object
                .call_method0(py, intern!(py, "pop"))
                .and_then(|o| o.extract(py))
        })
        .ok()
    }

    pub fn next(&mut self, mov: &Move) {
        Python::with_gil(|py| {
            mov.into_pyobject(py).map_err(PyErr::from).and_then(
                |m| self.python_object.call_method1(py, intern!(py, "push"), (m,))
            )
        })
        .unwrap();
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        Python::with_gil(|py| {
            let locals = [("board", &self.python_object)].into_py_dict(py)?;
            py.eval(c_str!("list(board.legal_moves)"), None, Some(&locals)).unwrap().extract()
        }).unwrap()
    }

    pub fn move_stack(&self) -> Vec<Move> {
        Python::with_gil(|py| {
            self.python_object
                .getattr(py, intern!(py, "move_stack"))
                .unwrap()
                .extract(py)
                .unwrap()
        })
    }
}

impl game::State for BoardState {
    type Step = Step;

    fn dup(&self) -> Self {
        Python::with_gil(|py| {
            return Self {
                python_object: self
                    .python_object
                    .call_method0(py, intern!(py, "copy"))
                    .unwrap(),
            };
        })
    }

    fn advance(&mut self, step: &Self::Step) {
        self.next(&step.0.unwrap());
    }

    //fn legal_moves(&self) -> Vec<Self::Step> {
    //    Python::with_gil(|py| {
    //        let locals = [("board", &self.python_object)].into_py_dict(py)?;
    //        let mov: Vec<Move> = py
    //            .eval(c_str!("list(board.legal_moves)"), None, Some(&locals))
    //            ?.extract()?;
    //        let turn: Color = self
    //            .python_object
    //            .getattr(py, intern!(py, "turn"))
    //            ?.extract::<i32>(py)?.into();
    //        Ok::<_, PyErr>(mov.into_iter().map(|m| (Some(m), !turn)).collect())
    //    }).unwrap()
    //}
}

impl BoardHistory {
    pub fn new(size: usize) -> Self {
        return BoardHistory {
            size: size,
            history: VecDeque::with_capacity(size),
        };
    }

    #[allow(dead_code)]
    pub fn push_front(&mut self, board: &Board) {
        if self.history.len() == self.size {
            self.history.pop_back();
        }
        self.history.push_front(board.clone());
    }

    #[allow(dead_code)]
    pub fn push_back(&mut self, board: &Board) {
        if self.history.len() < self.size {
            self.history.push_back(board.clone());
        }
    }

    pub fn view(&self, rotate: bool) -> Array3<i8> {
        let mut full = Array3::<i8>::zeros((8, 8, 14 * self.size));

        for idx in 0..self.history.len() {
            let board = self.history.get(idx).unwrap();
            let array = if rotate {
                board.rotate().encode_pieces()
            } else {
                board.encode_pieces()
            };
            let base = 14 * idx;
            full.slice_mut(s![.., .., base..base + 14]).assign(&array);
        }
        full
    }
}

fn _encode(node: &ArcRefNode<Step>, state: &BoardState) -> (Array3<i8>, Array3<i32>) {
    let current_board = state.to_board();
    let mut history = BoardHistory::new(LOOKBACK);
    let mut move_stack = state.move_stack();

    let mut cursor = Cursor::from_arc(node.clone());
    for _ in 0..LOOKBACK {
        // latter boards are kept at the front
        // must be in sync with encode_steps in lib.rs
        history.push_back(&get_board_from_moves(&move_stack));
        let (parent, last_move) = {
            let n = cursor.current();
            (n.parent.clone(), n.step.0)
        };
        match parent {
            None => break,
            Some(_) => {
                let m = move_stack.pop();
                assert!(m == last_move, "sanity check on the last move failed");
                cursor.navigate_up();
            }
        }
    }

    let turn = node.borrow().step.1;
    let encoded_boards = history.view(turn == Color::Black);
    let encoded_meta = current_board.encode_meta();

    (encoded_boards, encoded_meta)
}

fn _chess_predict(
    chess: &Chess,
    node: &ArcRefNode<Step>,
    state: &BoardState,
    argmax: bool,
) -> (Vec<Step>, Vec<f32>, f32) {
    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(node, state);
    let turn = node.borrow().step.1;

    let (moves_distr, score) = call_py_model(
        &chess.model,
        &chess.device,
        encoded_boards,
        encoded_meta,
        turn,
        &legal_moves,
    );

    let moves_distr = game::post_process_distr(moves_distr, argmax);
    let next_steps = legal_moves
        .into_iter()
        .map(|m| Step(Some(m), !turn))
        .collect();
    return (next_steps, moves_distr, score);
}

pub struct Chess {
    pub model: Py<PyAny>,
    pub device: String,
}

pub struct ChessTS {
    pub model: tch::CModule,
    pub device: tch::Device,
}

pub struct ChessEP {
    pub model: aotinductor::ModelPackage,
    pub device: tch::Device,
}

fn call_py_model(
    model: &Py<PyAny>,
    device: &str,
    boards: Array3<i8>,
    meta: Array3<i32>,
    turn: Color,
    steps: &Vec<Move>,
) -> (Vec<f32>, f32) {
    Python::with_gil(|py| {
        let encoded_boards = PyArray3::from_array(py, &boards);
        let encoded_meta = PyArray3::from_array(py, &meta);

        let code = c_str!(r#"
inp = np.concatenate((encoded_boards, encoded_meta), axis=-1).astype(np.float32)
inp = inp.transpose((2, 0, 1))
inp = torch.from_numpy(inp[np.newaxis, :])
with torch.no_grad():
    with torch.autocast(
        device_type=device, dtype=torch.float16, cache_enabled=False
    ):
        ret_distr, ret_score = model(inp.to(device))
ret_distr = ret_distr.detach().cpu().numpy().squeeze()
ret_distr = np.exp(ret_distr)
ret_score = ret_score.detach().cpu().item()
"#);
        let locals = [
            ("np", py.import("numpy").unwrap().as_ref()),
            ("torch", py.import("torch").unwrap().as_ref()),
            ("encoded_boards", encoded_boards.as_ref()),
            ("encoded_meta", encoded_meta.as_ref()),
            ("model", model.bind(py)),
            ("device", PyString::new(py, device).as_any()),
        ].into_py_dict(py)?;
        py.run(code, None, Some(&locals))?;

        let full_distr: Bound<'_, PyArray1<f32>> = locals
            .get_item("ret_distr")
            .unwrap()
            .unwrap()
            .downcast_into()?;

        // how good is the current board for the next player (turn)
        // This must be in sync with that in training script
        let score: f32 = locals
            .get_item("ret_score")
            .unwrap()
            .unwrap()
            .downcast()?
            .extract()?;

        // rotate if the next move is black
        let encoded_moves: Vec<i32> = steps
            .iter()
            .map(|m| {
                if turn == Color::Black {
                    m.rotate().encode()
                } else {
                    m.encode()
                }
            })
            .collect();

        let moves_distr: Vec<f32> = full_distr
            .readonly()
            .get_item(PyArray::from_vec(py, encoded_moves))
            .and_then(|o| o.extract())?;

        Ok::<_, PyErr>((moves_distr, score))
    }).unwrap()
}

impl Game<BoardState> for Chess {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        _chess_predict(self, node, state, argmax)
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

impl TchModel for ChessTS {
    fn forward(&self, inp: Tensor) -> (Tensor, Tensor) {
        let out = self.model.forward_is(&[tch::jit::IValue::from(inp)]).unwrap();
        <(Tensor, Tensor)>::try_from(out).unwrap()
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

impl Game<BoardState> for ChessTS {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

impl TchModel for ChessEP {
    fn forward(&self, inp: Tensor) -> (Tensor, Tensor) {
        let outs = self.model.run(&vec![inp]);
        assert!(outs.len() == 2);
        let mut iter = outs.into_iter();
        let r1 = iter.next().unwrap();
        let r2 = iter.next().unwrap();
        (r1, r2)
    }

    fn device(&self) -> tch::Device {
        self.device
    }
}

impl Game<BoardState> for ChessEP {
    fn predict(
        &self,
        node: &ArcRefNode<Step>,
        state: &BoardState,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        chess_tch_predict(self, node, state, argmax, false)
    }

    fn reverse_q(&self, node: &ArcRefNode<Step>) -> bool {
        // the chess model is expected to return the reward for the white player.
        node.borrow().step.1 == Color::Black
    }
}

fn _get_move_distribution(model_distr_output: Tensor, turn: Color, legal_moves: &Vec<Move>, return_full: bool) -> Vec<f32> {
    let full_distr = model_distr_output.to_dtype(tch::Kind::Float, true, false).to_device(tch::Device::Cpu);

    if return_full {
        Vec::<f32>::try_from(full_distr.squeeze().exp()).unwrap()
    }
    else {
        // rotate if the next move is black
        let encoded_moves: Vec<i64> = legal_moves
            .iter()
            .map(|m| {
                if turn == Color::Black {
                    m.rotate().encode() as i64
                } else {
                    m.encode() as i64
                }
            })
            .collect();
        let encoded_moves = Tensor::from_slice(&encoded_moves);
        Vec::<f32>::try_from(full_distr.take(&encoded_moves).exp()).unwrap()
    }
}

pub fn chess_tch_predict<M: TchModel>(
    chess: &M,
    node: &ArcRefNode<Step>,
    state: &BoardState,
    argmax: bool,
    return_full_distr: bool,
) -> (Vec<Step>, Vec<f32>, f32) {

    let legal_moves = state.legal_moves();

    if legal_moves.is_empty() {
        // the model always tell the score at the white side
        let outcome = match state.outcome().unwrap().winner {
            None => 0.0,
            Some(Color::White) => 1.0,
            Some(Color::Black) => -1.0,
        };
        return (Vec::new(), Vec::new(), outcome);
    }

    let (encoded_boards, encoded_meta) = _encode(&node, state);
    let turn = node.borrow().step.1;

    assert!(turn == state.turn());

    let inp = game::prepare_tensors(encoded_boards, encoded_meta, chess.device());
    let inp_debug = inp.alias_copy();

    let (full_distr, score) = chess.forward(inp);

    let score = game::tensor_to_f32(score).unwrap();

    if !score.is_finite() {
        let uuid = ShortUuid::generate();
        let filename = format!("/tmp/{}.tensor", uuid);
        inp_debug.save(std::path::Path::new(&filename)).unwrap();
        println!("Warning: score is bad {:?}", score);
        println!("input tensor is saved as {}", filename);
    }

    let moves_distr = _get_move_distribution(full_distr, turn, &legal_moves, return_full_distr);
    let moves_distr = game::post_process_distr(moves_distr, argmax);

    let next_steps = legal_moves
        .into_iter()
        .map(|m| Step(Some(m), !turn))
        .collect();

    return (next_steps, moves_distr, score);
}
