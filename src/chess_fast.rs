use core::convert::TryFrom;
use crate::chess::{BoardState, Color, Move, PieceType, Square};
use chess as rs_chess;


fn square_r(s: Square) -> rs_chess::Square {
    let rank = rs_chess::Rank::from_index(s.rank as usize);
    let file = rs_chess::File::from_index(s.file as usize);
    rs_chess::Square::make_square(rank, file)
}

fn piece_r(p: PieceType) -> rs_chess::Piece {
    match p {
        PieceType::Pawn => rs_chess::Piece::Pawn,
        PieceType::Knight => rs_chess::Piece::Knight,
        PieceType::Bishop => rs_chess::Piece::Bishop,
        PieceType::Rook => rs_chess::Piece::Rook,
        PieceType::Queen => rs_chess::Piece::Queen,
        PieceType::King => rs_chess::Piece::King,
    }
}

fn color_r(c: Color) -> rs_chess::Color {
    match c {
        Color::White => rs_chess::Color::White,
        Color::Black => rs_chess::Color::Black,
    }
}

#[allow(dead_code)]
fn move_r(c: &Move) -> rs_chess::ChessMove {
    rs_chess::ChessMove::new(
        square_r(c.from),
        square_r(c.to),
        c.promotion.map(piece_r),
    )
}

fn square_p(s: rs_chess::Square) -> Square {
    let rank = s.get_rank().to_index() as i32;
    let file = s.get_rank().to_index() as i32;
    Square {rank, file}
}

fn piece_p(p: rs_chess::Piece) -> PieceType {
    match p {
        rs_chess::Piece::Pawn => PieceType::Pawn,
        rs_chess::Piece::Knight => PieceType::Knight,
        rs_chess::Piece::Bishop => PieceType::Bishop,
        rs_chess::Piece::Rook => PieceType::Rook,
        rs_chess::Piece::Queen => PieceType::Queen,
        rs_chess::Piece::King => PieceType::King,
    }
}

#[allow(dead_code)]
fn color_p(c: rs_chess::Color) -> Color {
    match c {
        rs_chess::Color::White => Color::White,
        rs_chess::Color::Black => Color::Black,
    }
}

#[allow(dead_code)]
pub fn fast_legal_moves(board: &BoardState) -> Vec<Move> {
    let mut bb = rs_chess::BoardBuilder::new();
    let board_py = board.to_board();
    for (square, piece) in board_py.piece_map {
        bb.piece(square_r(square), piece_r(piece.piece_type), color_r(piece.color));
    }
    bb.side_to_move(color_r(board_py.turn));

    let board_rs = TryFrom::try_from(bb).unwrap();
    println!("{}", board_rs);
    rs_chess::MoveGen::new_legal(&board_rs).map(|m| {
        let from = square_p(m.get_source());
        let to = square_p(m.get_dest());
        println!("{} -> {}", from, to);
        let promotion = m.get_promotion().map(piece_p);
        Move { from, to, promotion, drop: None }
    }).collect()
}

#[test]
fn test_legal_moves() {
    use std::collections::HashSet;
    use std::hash::RandomState;

    let fen = "1k1r4/1r5p/p4n1P/1ppP1P2/PP6/4PP1b/3B4/R1N1K3 b - - 0 39";
    //"1k1r4/1r5p/p4n1P/1ppp1b2/PPP2P2/4PP1R/3B4/R1N1K3 w - - 0 38",
    //"8/k1B4p/2P2P1P/5P2/P7/8/1K6/2R5 w - - 3 58",
    //"8/2B4p/2P2P1P/Pk3P2/8/8/1KR5/8 w - - 1 60",
    //"r1kr4/2p4p/p4n1P/1p1p1b2/1P3PnR/3p1P2/P1PPN3/R1B2K2 b - - 7 30",
    let b = BoardState::from_fen(fen).unwrap();
    let m1 = HashSet::<_, RandomState>::from_iter(b.legal_moves());
    let m2 = HashSet::<_, RandomState>::from_iter(fast_legal_moves(&b));
    assert_eq!(m1, m2)
}

#[test]
fn test_1() {
    use std::str::FromStr;
    let fen = "1k1r4/1r5p/p4n1P/1ppP1P2/PP6/4PP1b/3B4/R1N1K3 b - - 0 39";
    let board = rs_chess::Board::from_str(fen).unwrap();
    rs_chess::MoveGen::new_legal(&board).map(|m| {
        let from = square_p(m.get_source());
        let to = square_p(m.get_dest());
        println!("{} -> {}", from, to);
    }).collect::<Vec<_>>();
}
