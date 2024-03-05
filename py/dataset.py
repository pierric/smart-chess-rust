import json

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

import libencoder


def _get_outcome(res):
    if res is None:
        return 0.

    if res["winner"] is None:
        return 0.

    if res["winner"] == "White":
        return 1.

    return -1.


class ChessDataset(Dataset):
    def __init__(self, trace_file):
        with open(trace_file, "r") as f:
            trace = json.load(f)

        self.outcome = _get_outcome(trace["outcome"])
        steps = [
            (chess.Move.from_uci(step[0]), [c[1] for c in step[2]])
            for step in trace["steps"]
        ]
        self.steps = libencoder.encode_steps(steps)


    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        boards, meta, dist = self.steps[idx][:3]
        inp = np.concatenate((boards, meta), axis=-1).astype(np.float32)
        inp = inp.transpose((2, 0, 1))
        turn = meta[0, 0, 0]
        assert idx % 2 == 1 - turn
        outcome = self.outcome * (1 if turn == 1 else -1)
        return torch.from_numpy(inp), torch.from_numpy(dist), torch.tensor([outcome], dtype=torch.float32)