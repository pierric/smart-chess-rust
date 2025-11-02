#! /bin/env python
import argparse
import time

import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trace", type=str, nargs="+", action="extend")
    parser.add_argument("--model", required=True, type=str)

    args = parser.parse_args()

    if not args.trace:
        print("No trace file specified.")
        return

    m = _load(args.model)

    ds = ConcatDataset([ChessDataset(f) for f in args.trace])

    print("Running inference...")
    time_list = []

    for inp, _, _ in tqdm(ds):
        inp = inp[None, ...].cuda().bfloat16()
        t1 = time.perf_counter()
        _ = m(inp)
        t2 = time.perf_counter()
        time_list.append(t2 - t1)

    time_list = np.array(time_list)
    print("mean: ", time_list.mean())
    print("std: ", time_list.std())


if __name__ == "__main__":
    main()
