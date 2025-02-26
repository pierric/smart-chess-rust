use std::cmp;
use std::collections::VecDeque;
use std::fmt;
use std::ops::Not;
use ndarray::{s, Array3, Ix3};
use serde::{Deserialize, Serialize, Serializer};
use serde::ser::SerializeMap;
use tch::Tensor;
use short_uuid::ShortUuid;
use pyo3::prelude::*;
use either::{Either, Left, Right};

use crate::{mcts, game};
use mcts::Cursor;
use game::{Game, TchModel, State};

pub const LOOKBACK: usize = 2;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Square {
    pub rank: i32,
    pub file: i32,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Move {
    pub from: Square,
    pub to: Square,
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Color {
    Black = 0,
    White,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize)]
pub struct Step(pub Option<Move>, pub Color);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Board {
    pub last_move: Option<Move>,
    pub turn: Color,
    pub pieces: [Option<Color>;9],
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
}

pub struct BoardHistory {
    pub size: usize,
    pub history: VecDeque<Board>,
}

#[derive(PartialOrd, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Outcome {
    Checkmate(Color),
    Stalemate,
    InsufficientMaterial,
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

impl<'a> FromPyObject<'a> for Color {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        return obj.extract::<i32>().map(|v| match v {
            0 => Color::Black,
            1 => Color::White,
            _ => panic!("bad value {} for color.", v)
        });
    }
}

impl<'a> IntoPyObject<'a> for Color {
    type Target = pyo3::types::PyBool;
    type Output = Bound<'a, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        let v: bool = self == Color::White;
        let obj = v.into_pyobject(py)?;
        Ok(obj.to_owned())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from.symbol(), self.to.symbol())
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

impl<'a> IntoPyObject<'a> for Step {
    type Target = pyo3::types::PyTuple;
    type Output = Bound<'a, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        (self.0, self.1).into_pyobject(py)
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

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chars = self.pieces.map(|p| {
            match (p, self.turn) {
                (Some(Color::White), Color::White) => "W", 
                (Some(Color::White), _) => "w", 
                (Some(Color::Black), Color::Black) => "B", 
                (Some(Color::Black), _) => "b", 
                _ => " ", 
            }
        });
        writeln!(f, "{}|{}|{}", chars[6], chars[7], chars[8])?;
        writeln!(f, "{}|{}|{}", chars[3], chars[4], chars[5])?;
        writeln!(f, "{}|{}|{}", chars[0], chars[1], chars[2])?;
        Ok(())
    }
}

impl Serialize for Outcome {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut res = serializer.serialize_map(Some(1))?;
        match self {
            Outcome::Checkmate(color) => { res.serialize_entry("winner", color)?; }
            _ => { res.serialize_entry("winner", &None::<Color>)?; }
        }
        res.end()
    }
}

const SQUARE_NAMES: [[&str; 3]; 3] = [
    ["a1", "b1", "c1"],
    ["a2", "b2", "c2"],
    ["a3", "b3", "c3"],
];

impl Square {
    pub fn symbol(&self) -> &str {
        SQUARE_NAMES[self.rank as usize][self.file as usize]
    }

    pub fn from_symbol(symbol: &str) -> Self {
        let repr = symbol.as_bytes();
        assert_eq!(repr.len(), 2);
        let file = (repr[0] - b'a') as i32;
        let rank = (repr[1] - b'1') as i32;
        Self{rank, file}
    }

    pub fn rotate(&self) -> Self {
        return Self {
            rank: 2 - self.rank,
            file: self.file,
        };
    }
}

impl Move {
    pub fn uci(&self) -> String {
        format!("{}{}", self.from.symbol(), self.to.symbol())
    }

    pub fn from_uci(uci: &str) -> Self {
        let from = Square::from_symbol(&uci[0..2]);
        let to = Square::from_symbol(&uci[2..]);
        Self {from, to}
    }

    pub fn rotate(&self) -> Self {
        let from = self.from.rotate();
        let to = self.to.rotate();
        return Self {
            from,
            to,
        };
    }

    pub fn encode(&self) -> i32 {
        let delta_file = self.to.file - self.from.file;
        let delta_rank = self.to.rank - self.from.rank;
        let direction_idx = match (delta_rank, delta_file) {
            (-1, -1) => 0,
            (-1, 0) => 1,
            (-1, 1) => 2,
            (1, -1) => 3,
            (1, 0) => 4,
            (1, 1) => 5,
            _ => panic!("impossible move: {} {}", delta_rank, delta_file),
        };

        self.from.rank * 3 * 6 + self.from.file * 6 + direction_idx
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

impl<'a> FromPyObject<'a> for Move {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> Result<Self, PyErr> {
        if obj.is_instance_of::<pyo3::types::PyString>() {
            let repr: String = FromPyObject::<'a>::extract_bound(obj)?;
            Ok(Move::from_uci(&repr))
        }
        else {
            use crate::chess;
            let cmov: chess::Move = FromPyObject::<'a>::extract_bound(obj)?;
            let from = Square{rank: cmov.from.rank, file: cmov.from.file};
            let to = Square{rank: cmov.to.rank, file: cmov.to.file};
            Ok(Move {from, to})
        }
    }
}

impl<'a> IntoPyObject<'a> for Move {
    type Target = pyo3::types::PyString;
    type Output = Bound<'a, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'a>) -> Result<Self::Output, Self::Error> {
        let repr = self.from.symbol().to_owned() + self.to.symbol();
        let obj = repr.into_pyobject(py)?;
        Ok(obj)
    }
}

impl Board {
    pub fn new() -> Self {
        let pieces = [
            Some(Color::White), Some(Color::White), Some(Color::White),
            None, None, None,
            Some(Color::Black), Some(Color::Black), Some(Color::Black),
        ];
        Self {
            last_move: None,
            turn: Color::White,
            pieces,
            halfmove_clock: 0,
            fullmove_number: 0,
        }
    }

    #[allow(dead_code)]
    pub fn from_str(repr: &str, turn: Color) -> Self {
        assert_eq!(repr.len(), 9);
        let repr = repr.as_bytes();
        let pieces: [Option<Color>; 9] = core::array::from_fn(|i| {
            let ch = repr[i];
            match ch {
                b'W' => Some(Color::White),
                b'B' => Some(Color::Black),
                b' ' => None,
                _ => panic!("bad char in the repr"),
            }
        });
        Self {
            last_move: None,
            turn,
            pieces,
            halfmove_clock: 0,
            fullmove_number: 0,
        }
    }

    pub fn get_piece(&self, rank: i32, file: i32) -> Option<Color> {
        let idx: usize = (rank * 3 + file).try_into().unwrap();
        self.pieces[idx]
    }

    pub fn set_piece(&mut self, rank: i32, file: i32, piece: Option<Color>) {
        let idx: usize = (rank * 3 + file).try_into().unwrap();
        self.pieces[idx] = piece;
    }

    pub fn turn(&self) -> Color {
        self.turn
    }

    pub fn possible_moves(&self) -> Vec<Move> {
        let mut moves = vec![];

        let direction = if self.turn == Color::White {1} else {-1};

        for (i, p) in self.pieces.iter().enumerate() {

            if *p != Some(self.turn) {
                continue;
            }

            let rank = i as i32 / 3;
            let file = i as i32 % 3;
            let from = Square{rank: rank, file: file};
            let rank1 = rank + direction;

            if rank1 < 0 || rank1 >= 3 {
                return moves;
            }

            if self.get_piece(rank1, file).is_none() {
                moves.push(Move{from, to: Square{rank: rank1, file: file}})
            }

            if file > 0 && self.get_piece(rank1, file-1) == Some(!self.turn) {
                moves.push(Move{from, to: Square{rank: rank1, file: file-1}})
            }

            if file < 2 && self.get_piece(rank1, file+1) == Some(!self.turn) {
                moves.push(Move{from, to: Square{rank: rank1, file: file+1}})
            }
        }

        moves
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        if self.outcome().is_some() {
            Vec::new()
        }
        else {
            self.possible_moves()
        }
    }

    pub fn step(&mut self, mov: &Move) {
        let src = self.get_piece(mov.from.rank, mov.from.file);
        let tgt = self.get_piece(mov.to.rank, mov.to.file);
        assert!(src == Some(self.turn) && tgt != Some(self.turn), "Invalid step {} for \n{}", mov, self);

        self.set_piece(mov.to.rank, mov.to.file, Some(self.turn));
        self.set_piece(mov.from.rank, mov.from.file, None);
        self.turn = !self.turn;
        self.halfmove_clock += 1;
    }

    pub fn outcome(&self) -> Option<Outcome> {
        for p in self.pieces[0..3].iter() {
            if *p == Some(Color::Black) {
                return Some(Outcome::Checkmate(Color::Black));
            }
        }

        for p in self.pieces[6..9].iter() {
            if *p == Some(Color::White) {
                return Some(Outcome::Checkmate(Color::White));
            }
        }

        if self.possible_moves().is_empty() {
            return Some(Outcome::Stalemate);
        }

        let mut cnt_white = 0;
        let mut cnt_black = 0;

        for p in self.pieces {
            match p {
                Some(Color::White) => {cnt_white += 1; },
                Some(Color::Black) => {cnt_black += 1; },
                None => {},
            }
        }

        if cnt_white == 0 || cnt_black == 0 {
            return Some(Outcome::InsufficientMaterial)
        }

        None
    }

    pub fn rotate(&self) -> Self {
        fn flip(p: Option<Color>) -> Option<Color> {
            p.map(|c| !c)
        }

        let pieces = [
            flip(self.pieces[6]), flip(self.pieces[7]), flip(self.pieces[8]),
            flip(self.pieces[3]), flip(self.pieces[4]), flip(self.pieces[5]),
            flip(self.pieces[0]), flip(self.pieces[1]), flip(self.pieces[2]),
        ];

        return Self {
            last_move: self.last_move.map(|m| m.rotate()),
            turn: self.turn,
            pieces,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
        };
    }

    pub fn encode_pieces(&self) -> Array3<i8> {
        let mut array = Array3::<i8>::zeros((3, 3, 2));

        for (idx, pos) in self.pieces.iter().enumerate() {
            match pos {
                None => {},
                Some(color) => {
                    let file = idx % 3;
                    let rank = idx / 3;
                    array[Ix3(rank, file, *color as usize)] = 1;
                }
            }
        }

        return array;
    }

    pub fn encode_meta(&self) -> Array3<i32> {
        let mut meta = Array3::<i32>::zeros((3, 3, 2));
        meta.slice_mut(s![.., .., 0]).fill(self.turn as i32);
        meta.slice_mut(s![.., .., 1]).fill(self.halfmove_clock as i32);
        return meta;
    }
}

impl game::State for Board {
    type Step = Step;

    fn dup(&self) -> Self {
        self.clone()
    }

    fn advance(&mut self, step: &Self::Step) {
        // step.1 is the turn after taking the step
        // therefore the opponent's color of the current state
        assert!(step.0.is_some() && step.1 == !self.turn, "{:?} {}", step, self.turn);
        self.step(&step.0.unwrap());
    }
}

impl BoardHistory {
    pub fn new(size: usize) -> Self {
        return BoardHistory {
            size: size,
            history: VecDeque::with_capacity(size),
        };
    }

    #[allow(dead_code)]
    pub fn push_front(&mut self, board: Board) {
        if self.history.len() == self.size {
            self.history.pop_back();
        }
        self.history.push_front(board);
    }

    #[allow(dead_code)]
    pub fn push_back(&mut self, board: Board) {
        if self.history.len() < self.size {
            self.history.push_back(board);
        }
    }

    pub fn view(&self, rotate: bool) -> Array3<i8> {
        let mut full = Array3::<i8>::zeros((3, 3, 2 * self.size));

        for idx in 0..self.history.len() {
            let board = self.history.get(idx).unwrap();
            let array = if rotate {
                board.rotate().encode_pieces()
            } else {
                board.encode_pieces()
            };
            let base = 2 * idx;
            full.slice_mut(s![.., .., base..base + 2]).assign(&array);
        }
        full
    }
}

pub struct ChessTS {
    pub model: tch::CModule,
    pub device: tch::Device,
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

pub struct ChessEP {
    pub model: aotinductor::ModelPackage,
    pub device: tch::Device,
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

fn _encode(node: &mcts::ArcRefNode<Step>, state: &Board) -> (Array3<i8>, Array3<i32>) {
    let mut history = BoardHistory::new(LOOKBACK);

    let mut steps: Vec<Move> = vec![];
    let mut cursor = Cursor::from_arc(node.clone());

    let mut mov = cursor.current().step.0;
    while let Some(m) = mov {
        steps.push(m);
        cursor.navigate_up();
        mov = cursor.current().step.0;
    }
    steps.reverse();

    let start = cmp::max(steps.len(), LOOKBACK) - LOOKBACK;
    let mut board = Board::new();
    history.push_front(board.clone());

    for m in &steps[0..start] {
        board.step(m);
    }

    for m in &steps[start..] {
        board.step(m);
        history.push_front(board.clone());
    }

    let turn = node.borrow().step.1;
    let encoded_boards = history.view(turn == Color::Black);

    assert!(board == *state);

    let encoded_meta = state.encode_meta();
    (encoded_boards, encoded_meta)
}

fn _prepare_tensors(boards: Array3<i8>, meta: Array3<i32>, device: tch::Device) -> Tensor {
    // copy to device in sync mode
    // otherwise may cause corruption in data (mps)
    let encoded_boards =
        Tensor::try_from(boards)
            .unwrap()
            .to_device_(device, tch::Kind::BFloat16, false, false);
    let encoded_meta =
        Tensor::try_from(meta)
            .unwrap()
            .to_device_(device, tch::Kind::BFloat16, false, false);

    Tensor::cat(&[encoded_boards, encoded_meta], 2)
        .permute([2, 0, 1])
        .contiguous() // aot model expects contiguous tensor
        .unsqueeze(0)
}

fn _get_move_distribution(model_distr_output: Tensor, turn: Color, next_steps: &Vec<Step>, return_full: bool) -> Vec<f32> {
    let full_distr = model_distr_output.to_dtype(tch::Kind::Float, true, false).to_device(tch::Device::Cpu);

    if return_full {
        Vec::<f32>::try_from(full_distr.squeeze().exp()).unwrap()
    }
    else {
        // rotate if the next move is black
        let encoded_moves: Vec<i64> = next_steps
            .iter()
            .map(|step| {
                let m = step.0.unwrap();
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


pub fn _predict_raw_common<M: TchModel>(
    m: &M, node: &mcts::ArcRefNode<Step>, state: &Board
) -> Either<(Vec<Step>, Tensor, f32), f32> {

        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            // the model always tell the score at the white side
            let outcome = match state.outcome() {
                Some(Outcome::Checkmate(Color::White)) => 1.0,
                Some(Outcome::Checkmate(Color::Black)) => -1.0,
                _ => 0.0,
            };
            return Right(outcome);
        }

        let (encoded_boards, encoded_meta) = _encode(&node, state);
        let turn = node.borrow().step.1;

        assert!(turn == state.turn(), "{} {}", turn, state.turn);

        let inp = _prepare_tensors(encoded_boards, encoded_meta, m.device());
        let inp_debug = inp.alias_copy();

        let (full_distr, score) = m.forward(inp);

        let score = game::tensor_to_f32(score).unwrap();
        if !score.is_finite() {
            let uuid = ShortUuid::generate();
            let filename = format!("/tmp/{}.tensor", uuid);
            inp_debug.save(std::path::Path::new(&filename)).unwrap();
            println!("Warning: score is bad {:?}", score);
            println!("input tensor is saved as {}", filename);
        }

        let next_steps = legal_moves
            .into_iter()
            .map(|m| Step(Some(m), !turn))
            .collect();

        Left((next_steps, full_distr, score))
}

impl Game<Board> for ChessTS {

    fn predict(
        &self,
        node: &mcts::ArcRefNode<Step>,
        state: &Board,
        argmax: bool,
    ) -> (Vec<Step>, Vec<f32>, f32) {
        match _predict_raw_common(self, node, state) {
            Right(outcome) => (vec![], vec![], outcome),
            Left((next_steps, full_distr, score)) => {
                let turn = node.borrow().step.1;
                let moves_distr = _get_move_distribution(full_distr, turn, &next_steps, false);
                let moves_distr = game::post_process_distr(moves_distr, argmax);
                (next_steps, moves_distr, score)
            },
        }
    }

    fn reverse_q(&self, node: &mcts::ArcRefNode<Step>) -> bool {
        node.borrow().step.1 == Color::Black
    }
}

impl Game<Board> for ChessEP {

    fn predict(
        &self,
        node: &mcts::ArcRefNode<Step>,
        state: &Board,
        argmax: bool,
    ) -> (Vec<<Board as State>::Step>, Vec<f32>, f32) {
        match _predict_raw_common(self, node, state) {
            Right(outcome) => (vec![], vec![], outcome),
            Left((next_steps, full_distr, score)) => {
                let turn = node.borrow().step.1;
                let moves_distr = _get_move_distribution(full_distr, turn, &next_steps, false);
                let moves_distr = game::post_process_distr(moves_distr, argmax);
                (next_steps, moves_distr, score)
            },
        }
    }

    fn reverse_q(&self, node: &mcts::ArcRefNode<Step>) -> bool {
        node.borrow().step.1 == Color::Black
    }
}

#[test]
fn test_board_new() {
    let b = Board::new();
    assert_eq!(b.turn, Color::White);
    assert_eq!(b.pieces[0], Some(Color::White));
    assert_eq!(b.pieces[3], None);
    assert_eq!(b.pieces[6], Some(Color::Black));
}

#[test]
fn test_legal_moves() {
    use std::collections::HashSet;
    let b = Board::new();
    let m = b.legal_moves();

    let a: HashSet<Move> = HashSet::from_iter(m.into_iter());
    let m = (0..3).map(|f| {
        Move{from: Square{rank:0, file:f}, to: Square{rank:1, file:f}}
    });
    let b: HashSet<Move> = HashSet::from_iter(m);
    assert_eq!(a, b);
}

#[test]
fn test_board_from_str() {
    let b = Board::from_str("WW BBW  B", Color::White);
    assert_eq!(b.pieces[0], Some(Color::White));
    assert_eq!(b.pieces[3], Some(Color::Black));
    assert_eq!(b.pieces[6], None);
}

#[test]
fn test_legal_moves_attack_w() {
    use std::collections::HashSet;
    let b = Board::from_str("WW BBW  B", Color::White);
    let m = b.legal_moves();
    let m0 = Move{from: Square{rank:0, file:0}, to: Square{rank:1, file:1}};
    let m1 = Move{from: Square{rank:0, file:1}, to: Square{rank:1, file:0}};
    assert_eq!(m.len(), 2, "{:?}", m);

    let a: HashSet<Move> = HashSet::from_iter(m.into_iter());
    let b: HashSet<Move> = HashSet::from([m0, m1]);
    assert_eq!(a, b);
}

#[test]
fn test_legal_moves_attack_b() {
    use std::collections::HashSet;
    let b = Board::from_str("WW BBW  B", Color::Black);
    let m = b.legal_moves();
    let m0 = Move{from: Square{rank:1, file:0}, to: Square{rank:0, file:1}};
    let m1 = Move{from: Square{rank:1, file:1}, to: Square{rank:0, file:0}};
    assert_eq!(m.len(), 2, "{:?}", m);

    let a: HashSet<Move> = HashSet::from_iter(m.into_iter());
    let b: HashSet<Move> = HashSet::from([m0, m1]);
    assert_eq!(a, b);
}

#[test]
fn test_outcome_white() {
    let b = Board::from_str("W    BW B", Color::White);
    assert_eq!(b.outcome(), Some(Outcome::Checkmate(Color::White)));

    let b = Board::from_str("  W BB W ", Color::Black);
    assert_eq!(b.outcome(), Some(Outcome::Checkmate(Color::White)));
}

#[test]
fn test_outcome_black() {
    let b = Board::from_str("WB  WB  B", Color::Black);
    assert_eq!(b.outcome(), Some(Outcome::Checkmate(Color::Black)));

    let b = Board::from_str(" B W  B B", Color::Black);
    assert_eq!(b.outcome(), Some(Outcome::Checkmate(Color::Black)));
}

#[test]
fn test_rotate_board() {
    let b = Board::from_str("   BBBWWW", Color::Black);
    let r = format!("{}", b.rotate());
    assert_eq!(r, " | | \nw|w|w\nB|B|B\n");

    let arr = b.encode_pieces();
    assert_eq!(arr.shape(), [3,3,2]);
    assert_eq!(arr.as_slice(), Some([
        0,0,0,0,0,0,
        1,0,1,0,1,0,
        0,1,0,1,0,1,
    ].as_slice()));

    let arr = b.rotate().encode_pieces();
    assert_eq!(arr.shape(), [3,3,2]);
    assert_eq!(arr.as_slice(), Some([
        1,0,1,0,1,0,
        0,1,0,1,0,1,
        0,0,0,0,0,0,
    ].as_slice()));
}

#[test]
fn test_rotate_move() {
    let m = Move::from_uci("a3a2").rotate();
    assert_eq!(m.uci(), "a1a2");

    let m = Move::from_uci("c2b1").rotate();
    assert_eq!(m.uci(), "c2b3");
}
