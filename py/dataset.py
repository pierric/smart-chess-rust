import io
import json

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import libsmartchess


def _get_outcome(res):
    if res is None:
        return 0.0

    if res["winner"] is None:
        return 0.0

    if res["winner"] == "White":
        return 1.0

    return -1.0


def _prepare(boards, meta, dist, outcome):
    inp = np.concatenate((boards, meta), axis=-1).astype(np.float32)
    inp = inp.transpose((2, 0, 1))
    # turn = meta[0, 0, 0]
    # if turn == 0:
    #    outcome = -outcome
    return (
        torch.from_numpy(inp),
        torch.from_numpy(dist),
        torch.tensor([outcome], dtype=torch.float32),
    )


class ChessDataset(Dataset):
    def __init__(self, trace_file, apply_mirror=False):
        with open(trace_file, "r") as f:
            trace = json.load(f)

        self.outcome = _get_outcome(trace["outcome"])
        steps = [
            (chess.Move.from_uci(step[0]), [c[1] for c in step[2]])
            for step in trace["steps"]
        ]
        self.steps = libsmartchess.encode_steps(steps, apply_mirror)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        boards, meta, dist = self.steps[idx][:3]
        return _prepare(boards, meta, dist, self.outcome)


class ValidationDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file).iloc[:10]
        self.plays = [self._encode(p) for p in df.moves]
        self.indices = np.concatenate(
            [[i] * n for i, n in enumerate(map(len, self.plays))]
        )

        self.total_steps = np.cumsum(list(map(len, self.plays)))

        outcome = {
            "white": 1,
            "black": -1,
            "draw": 0,
        }
        self.outcomes = df.winner.apply(lambda s: outcome[s])

    def _encode(self, pgn):
        game = chess.pgn.read_game(io.StringIO(pgn))
        node = game.next()
        steps = []

        while node:
            move = node.move
            legal_moves = list(node.board().legal_moves)
            num_acts = [0] * len(legal_moves)
            node = node.next()
            if node is None:
                break

            num_acts[legal_moves.index(node.move)] = 1
            steps.append((move, num_acts))

        return libsmartchess.encode_steps(steps, False)

    def __len__(self):
        return self.total_steps[-1]

    def __getitem__(self, idx):
        play_idx = self.indices[idx]
        base = 0 if 0 == play_idx else self.total_steps[play_idx - 1]
        local_step = idx - base
        boards, meta, dist = self.plays[play_idx][local_step][:3]
        return _prepare(boards, meta, dist, self.outcomes[play_idx])
