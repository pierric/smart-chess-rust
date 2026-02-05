#!/usr/bin/env python
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset
import numpy as np

from dataset import ChessDataset
from module import load_model_for_inference


def _load(model_spec):
    parts = model_spec.split(":", 1)
    n_res_blocks = None
    if len(parts) > 1:
        n_res_blocks, checkpoint = parts
        n_res_blocks = int(n_res_blocks)
    else:
        checkpoint = parts[0]

    return load_model_for_inference(checkpoint, n_res_blocks=n_res_blocks)


def total_variation_distance(p1, p2):
    return abs(p1 - p2).sum() / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace", type=str, nargs="+", action="extend")
    parser.add_argument("--model1", required=True, type=str)
    parser.add_argument("--model2", required=True, type=str)

    args = parser.parse_args()

    if not args.trace:
        print("No trace file specified.")
        return

    m1 = _load(args.model1)
    m2 = _load(args.model2)

    ds = ConcatDataset([ChessDataset(f) for f in args.trace])

    differences_policy = []
    differences_value = []

    print("Running inference...")

    for inp, meta, _, _ in tqdm(ds):
        inp = inp[None, ...].cuda()
        meta = meta[None, ...].cuda()
        inp_f16 = inp.half()
        meta_f16 = meta.half()
        p1, d1 = m1(inp_f16, meta_f16)
        p2, d2 = m2(inp_f16, meta_f16)
        differences_policy.append(
            total_variation_distance(p1.exp().cpu().numpy(), p2.exp().cpu().numpy())
        )
        differences_value.append(abs(d1.cpu() - d2.cpu()).item())

    diff = np.array(differences_policy)
    print(
        "policy difference:",
        {
            "mean": diff.mean(),
            "std": diff.std(),
            "max": diff.max(),
            "min": diff.min(),
        },
    )

    diff = np.array(differences_value)
    print(
        "value difference:",
        {
            "mean": diff.mean(),
            "std": diff.std(),
            "max": diff.max(),
            "min": diff.min(),
        },
    )


if __name__ == "__main__":
    main()
