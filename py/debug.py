import numpy as np
import chess


def decode_pieces(arr, color):
    types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    all_pieces = []

    for idx, typ in enumerate(types):
        rs, fs = np.where(arr[idx] == 1)
        squares = [chess.square(file, rank) for file, rank in zip(fs, rs)]
        piece = chess.Piece(typ, color)
        all_pieces.extend([(sq, piece, False) for sq in squares])

    return all_pieces


def decode_board(arr):
    if np.array_equal(arr[:12], 0):
        return None

    pieces_white = decode_pieces(arr[0:6, :, :], chess.WHITE)
    pieces_black = decode_pieces(arr[6:12, :, :], chess.BLACK)

    board = chess.Board(fen=None)
    for square, piece, promoted in pieces_white + pieces_black:
        board.set_piece_at(square, piece, promoted)

    return board
