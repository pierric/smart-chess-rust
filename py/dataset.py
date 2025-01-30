import io
import json

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset


import libsmartchess


def _get_outcome(res):
    if res is None:
        return 0.0

    if res["winner"] is None:
        return 0.0

    if res["winner"] == "White":
        return 1.0

    if res["winner"] == "Black":
        return -1.0

    raise RuntimeError(f"Unknown outcome: {res}")


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
        if isinstance(trace_file, dict):
            trace = trace_file

        elif isinstance(trace_file, io.TextIOBase):
            trace = json.load(trace_file)

        else:
            assert isinstance(trace_file, str)
            with open(trace_file, "r") as f:
                trace = json.load(f)

        self.outcome = _get_outcome(trace["outcome"])

        # steps in the trace files are [(move, score, [sibling moves])]
        # libsmartchess.encode_steps will use the [slibing_moves] to
        # encode a distribution of possibile moves of the current state
        # ** AND THEN ** make the move.
        trace_steps = trace["steps"]
        steps = [
            (
                chess.Move.from_uci(step[0]),
                [(chess.Move.from_uci(c[0]), c[1]) for c in step[2]],
            )
            for step in trace_steps
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
        traces = [self._to_trace(m, w) for m, w in zip(df.moves, df.winner)]
        self.dataset = ConcatDataset([ChessDataset(t) for t in traces])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _to_trace(self, pgn, winner):
        node = chess.pgn.read_game(io.StringIO(pgn))
        steps = []

        while node:
            num_acts = {m: 0 for m in node.board().legal_moves}
            node = node.next()

            if node is None:
                break

            move = node.move
            num_acts[node.move] = 1
            steps.append(
                (move.uci(), 0.0, [(m.uci(), c, 0.0) for m, c in num_acts.items()])
            )

        outcome = {
            "white": "White",
            "black": "Black",
            "draw": None,
        }

        return {
            "outcome": {"winner": outcome[winner]},
            "steps": steps,
        }


class ValidationDataset2(Dataset):
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
        node = chess.pgn.read_game(io.StringIO(pgn))
        steps = []

        while node:
            num_acts = {m: 0 for m in node.board().legal_moves}

            node = node.next()
            if node is None:
                break

            move = node.move
            num_acts[node.move] = 1
            steps.append((move, list(num_acts.items())))

        return libsmartchess.encode_steps(steps, False)

    def __len__(self):
        return self.total_steps[-1]

    def __getitem__(self, idx):
        play_idx = self.indices[idx]
        base = 0 if 0 == play_idx else self.total_steps[play_idx - 1]
        local_step = idx - base
        boards, meta, dist = self.plays[play_idx][local_step][:3]
        return _prepare(boards, meta, dist, self.outcomes[play_idx])
