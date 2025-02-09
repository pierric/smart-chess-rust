#!/usr/bin/env python
import argparse
import os
from functools import partial

import nn

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

# segfault on GH system, debug compile helps to avoid it.
# os.environ["AOT_INDUCTOR_DEBUG_COMPILE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", nargs="*", action="extend")
    parser.add_argument(
        "-m", "--mode", choices=["bf16", "amp", "ptq", "simple"], default="simple"
    )
    parser.add_argument("-f", "--format", choices=["onnx", "pt", "pt2"], default="pt2")
    parser.add_argument("-n", "--n-res-blocks", type=int, required=True)
    parser.add_argument("--calib", nargs="*")
    args = parser.parse_args()

    routing = {
        ("simple", "pt"): lambda: nn.export,
        ("amp", "pt"): lambda: nn.export_fp16,
        ("bf16", "pt"): lambda: nn.export_bf16,
        ("bf16", "pt2"): lambda: nn.export_pt2_bf16,
        ("ptq", "onnx"): lambda: partial(nn.export_ptq, calib=args.calib),
    }

    func = routing[(args.mode, args.format)]()

    for path in args.checkpoint:
        stamm, _ = os.path.splitext(path)
        func(args.n_res_blocks, checkpoint=path, output=f"{stamm}.{args.format}")


if __name__ == "__main__":
    main()
