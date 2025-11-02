import json
import argparse
from pathlib import Path
import torch
import chess
import libsmartchess as libsc
from module import load_model_for_inference


class RsGame:
    def __init__(self, checkpoint):
        self.game = libsc.chess_play_new(checkpoint, "cuda", [])

    def __call__(self, mov):
        libsc.chess_play_apply_move(self.game, mov)
        return libsc.chess_play_inference(self.game)

    def get_encoded(self):
        return libsc.chess_play_encode(self.game)


def total_variation_distance(p1, p2):
    return abs(p1 - p2).sum() / 2


def process_trace(trace, model_onnx, py_model):
    with open(trace, "r") as f:
        trace = json.load(f)

    trace_steps = trace["steps"]
    steps = [chess.Move.from_uci(step[0]) for step in trace_steps]

    total_score_diff = 0
    total_distr_diff = 0

    game = RsGame(model_onnx)
    for mov in steps:
        rs_steps, rs_prior, rs_outcome = game(mov)

        encoded_moves = [libsc.chess_encode_move(not t, m) for m, t in rs_steps]

        encoded_board, encoded_meta = game.get_encoded()
        encoded_board = (
            torch.from_numpy(encoded_board[None, ...])
            .permute(0, 3, 1, 2)
            .cuda()
            .bfloat16()
        )
        encoded_meta = torch.from_numpy(encoded_meta[None, ...]).cuda().bfloat16()
        policy_out, value_out = py_model(encoded_board, encoded_meta)
        value_out = value_out.item()
        total_score_diff += abs(rs_outcome - value_out)

        if encoded_moves:
            move_indices = torch.tensor(encoded_moves)
            move_distr = torch.take(policy_out.cpu(), move_indices).exp()
            move_distr = move_distr / (move_distr.sum() + 1e-5)
            total_distr_diff += total_variation_distance(
                torch.tensor(rs_prior), move_distr
            )

    print("avg score difference: ", total_score_diff / len(steps))
    print("avg distr difference: ", total_distr_diff / len(steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace", type=str, nargs="+", action="extend")
    parser.add_argument("--model", required=True, type=str)

    args = parser.parse_args()

    if not args.trace:
        print("No trace file specified.")
        return

    model = Path(args.model)
    assert model.exists(), f"{model} doesn't exist"

    ckpt = model.parent / (model.stem + ".ckpt")
    assert ckpt.exists(), f"{ckpt} doesn't exist"

    py_model = load_model_for_inference(str(ckpt), 16)

    for t in args.trace:
        process_trace(t, args.model, py_model)


if __name__ == "__main__":
    main()
