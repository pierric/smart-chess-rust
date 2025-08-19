import tempfile
import os
from contextlib import contextmanager

import torch


def export_ptq(model, *, output, calib):
    import pytorch_quantization
    import pytorch_quantization.quant_modules
    from itertools import islice
    from dataset import ChessDataset
    from torch.utils.data import ConcatDataset, DataLoader

    pytorch_quantization.quant_modules.initialize()

    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            module.enable_calib()
            module.disable_quant()

    dss = ConcatDataset([ChessDataset(trace_file) for trace_file in calib])
    train_loader = DataLoader(
        dss, num_workers=4, batch_size=1, shuffle=True, drop_last=True
    )
    for example in islice(train_loader, 50):
        model(example[0])

    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            module.load_calib_amax()
            module.disable_calib()
            module.enable_quant()

    model = model.cuda()

    dummy_input = torch.randn(1, 119, 8, 8, dtype=torch.float32, device="cuda")

    with pytorch_quantization.enable_nonnx_export():
        # enable_onnx_checker needs to be disabled. See notes below.
        torch.onnx.export(
            model,
            dummy_input,
            output,
            # verbose=True,
            input_names=["inp"],
            output_names=["policy", "value"],
        )

    #
    ## NOT possible to export to a torchscript
    #
    # model_jit = torch.jit.trace(model, [dummy_input])
    # torch.jit.save(model_jit, output)


def export_fp16(model, *, output):
    # assuming the model was trained with AMP
    x = torch.randn(1, 119, 8, 8, dtype=torch.float32).cuda()

    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, cache_enabled=False
        ):
            model_jit = torch.jit.trace(model, [x])
            # model_jit = torch.jit.freeze(model_jit)
            model_jit = torch.jit.optimize_for_inference(model_jit)

    # model_jit = torch.compile(model_jit, mode="reduce-overhead", fullgraph=True)

    # torch_tensorrt compiled model is 20% faster
    # though it requires the libtorchtrt.so being loaded beforehand
    # compiled = torch_tensorrt.compile(
    #     model_jit,
    #     inputs=[torch_tensorrt.Input((1, 119, 8, 8))],
    #     enabled_precisions=[torch.float, torch.half],
    #     ir="torchscript",
    # )

    torch.jit.save(model_jit, output)


def export_pt_f32(model, *, inp_shape, device, output):
    x = torch.randn(1, *inp_shape).to(device=device)

    with torch.no_grad():
        model_jit = torch.jit.trace(model, [x])
        model_jit = torch.jit.optimize_for_inference(model_jit)

    torch.jit.save(model_jit, output)


def export_pt_bf16(model, *, inp_shape, device, output):
    # assuming the model was trained with AMP

    # model_jit = torch.jit.script(model)
    # model_jit = torch.jit.optimize_for_inference(model_jit)

    x = torch.randn(1, *inp_shape).to(device=device)

    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, cache_enabled=False
        ):
            model_jit = torch.jit.trace(model, [x])
            model_jit = torch.jit.optimize_for_inference(model_jit)

    torch.jit.save(model_jit, output)


def export_pt2_bf16(model, *, inp_shape, device, output):
    inductor_configs = {
        "conv_1x1_as_mm": True,
        "epilogue_fusion": False,
        "coordinate_descent_tuning": True,
        "coordinate_descent_check_all_directions": True,
        "max_autotune": True,
        "triton.cudagraphs": True,
    }

    torch.set_float32_matmul_precision("high")
    # assuming the model was trained with AMP

    # fix the batch dim to be 1. There is some wierd restriction
    # with torch.export.Dim, the batch dim can be anything other than 1
    x = torch.randn(2, *inp_shape, dtype=torch.bfloat16).to(device=device)
    batch = torch.export.Dim("batch")

    with torch.no_grad():
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, cache_enabled=False
        ):
            ep = torch.export.export(
                model, args=(x,), dynamic_shapes=({0: batch},), strict=True
            )
            # torch.export.save(ep, "/tmp/exported.pt2")
            torch._inductor.aoti_compile_and_package(
                ep, package_path=output, inductor_configs=inductor_configs
            )


@contextmanager
def unlinking(tmpfile):
    fd, path = tmpfile
    os.close(fd)
    try:
        yield path
    finally:
        os.unlink(path)


def export_onnx(model, *, inp_shape, device, output, fp16):
    x = torch.randn(1, *inp_shape, dtype=torch.float32).to(device=device)

    with unlinking(tempfile.mkstemp(suffix=".onnx")) as tmp:
        with torch.no_grad():
            torch.onnx.export(
                model, x, tmp, input_names=["inp"], output_names=["policy", "value"]
            )

        if not fp16:
            import shutil

            shutil.copyfile(tmp, output)
            return

        import onnx
        from onnxconverter_common import auto_convert_mixed_precision

        model = onnx.load(tmp)
        model = auto_convert_mixed_precision(
            model, {"inp": x.cpu().numpy()}, rtol=0.01, atol=0.001, keep_io_types=True
        )
        onnx.save(model, output)
