#!/usr/bin/env python
import argparse
import os
from functools import partial

import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", nargs="*", action="extend")
    parser.add_argument(
        "-m", "--mode", choices=["amp", "ptq", "simple"], default="simple"
    )
    parser.add_argument("-n", "--n-res-blocks", type=int, required=True)
    parser.add_argument("--calib", nargs="*")
    args = parser.parse_args()

    func = None
    ext = ".pt"
    if args.mode == "simple":
        func = nn.export
    elif args.mode == "amp":
        func = nn.export_fp16
    elif args.mode == "ptq":
        func = partial(nn.export_ptq, calib=args.calib)
        ext = ".onnx"

    for path in args.checkpoint:
        stamm, _ = os.path.splitext(path)
        func(args.n_res_blocks, checkpoint=path, output=f"{stamm}{ext}")


if __name__ == "__main__":
    main()
