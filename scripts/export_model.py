#!/usr/bin/env python
import argparse
import os
from functools import partial

import module
import export

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

# segfault on GH system, debug compile helps to avoid it.
# os.environ["AOT_INDUCTOR_DEBUG_COMPILE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument(
        "-m", "--mode", choices=["bf16", "fp16", "ptq", "simple"], default="simple"
    )
    parser.add_argument("-f", "--format", choices=["onnx", "pt", "pt2"], default="pt2")
    parser.add_argument("-n", "--n-res-blocks", type=int, required=False)
    parser.add_argument("-d", "--device", choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--calib", nargs="*")
    args = parser.parse_args()

    assert args.n_res_blocks is not None
    from module import load_model

    model = load_model(
        n_res_blocks=args.n_res_blocks,
        checkpoint=args.checkpoint,
        device=args.device,
        inference=True,
        compile=False,
    )
    shape1 = (112, 8, 8)
    shape2 = (7,)

    routing = {
        ("simple", "pt"): lambda: export.export,
        ("fp16", "pt"): lambda: export.export_fp16,
        ("bf16", "pt"): lambda: export.export_pt_bf16,
        ("bf16", "pt2"): lambda: export.export_pt2_bf16,
        ("simple", "onnx"): lambda: partial(export.export_onnx, fp16=False),
        ("fp16", "onnx"): lambda: partial(export.export_onnx, fp16=True),
        ("ptq", "onnx"): lambda: partial(export.export_ptq, calib=args.calib),
    }

    stamm, _ = os.path.splitext(args.checkpoint)
    func = routing[(args.mode, args.format)]()
    func(
        model,
        inp_shapes=[shape1, shape2],
        device=args.device,
        output=f"{stamm}.{args.format}",
    )


if __name__ == "__main__":
    main()
