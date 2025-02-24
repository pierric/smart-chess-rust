#!/usr/bin/env python
import argparse
import os
from functools import partial

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

# segfault on GH system, debug compile helps to avoid it.
# os.environ["AOT_INDUCTOR_DEBUG_COMPILE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument(
        "-m", "--mode", choices=["bf16", "amp", "ptq", "simple"], default="simple"
    )
    parser.add_argument("-f", "--format", choices=["onnx", "pt", "pt2"], default="pt2")
    parser.add_argument("-n", "--n-res-blocks", type=int, required=False)
    parser.add_argument("-g", "--game", choices=["chess", "hexapawn"], default="chess")
    parser.add_argument("-d", "--device", choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--calib", nargs="*")
    args = parser.parse_args()

    if args.game == "chess":
        assert args.n_res_blocks is not None
        from game_chess.module import load_model
        model = load_model(
            n_res_blocks=args.n_res_blocks,
            checkpoint=args.checkpoint,
            device=args.device,
            inference=True,
            compile=False,
        )
        inp_shape = (119, 8, 8)

    elif args.game == "hexapawn":
        from game_hexapawn.module import load_model
        model = load_model(
            checkpoint=args.checkpoint,
            device=args.device,
            inference=True,
            compile=False,
        )
        inp_shape = (11, 3, 3)

    import export
    routing = {
        ("simple", "pt"): lambda: export.export,
        ("amp", "pt"): lambda: export.export_fp16,
        ("bf16", "pt"): lambda: export.export_pt_bf16,
        ("bf16", "pt2"): lambda: export.export_pt2_bf16,
        ("ptq", "onnx"): lambda: partial(export.export_ptq, calib=args.calib),
    }

    stamm, _ = os.path.splitext(args.checkpoint)
    func = routing[(args.mode, args.format)]()
    func(model,inp_shape=inp_shape, device=args.device, output=f"{stamm}.{args.format}")


if __name__ == "__main__":
    main()
