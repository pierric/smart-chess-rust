import tempfile
import os
from contextlib import contextmanager

import torch


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


def export_pt_f32(model, *, inp_shapes, device, output):
    inp1 = torch.randn(1, *inp_shapes[0], dtype=torch.float32).to(device=device)
    inp2 = torch.randn(1, *inp_shapes[1], dtype=torch.float32).to(device=device)

    with torch.no_grad():
        model_jit = torch.jit.trace(model, [inp1, inp2])
        model_jit = torch.jit.optimize_for_inference(model_jit)

    torch.jit.save(model_jit, output)


def export_pt_bf16(model, *, inp_shapes, device, output):
    # assuming the model was trained with AMP

    # model_jit = torch.jit.script(model)
    # model_jit = torch.jit.optimize_for_inference(model_jit)

    inp1 = torch.randn(1, *inp_shapes[0], dtype=torch.float32).to(device=device)
    inp2 = torch.randn(1, *inp_shapes[1], dtype=torch.float32).to(device=device)

    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, cache_enabled=False
        ):
            model_jit = torch.jit.trace(model, [inp1, inp2])
            model_jit = torch.jit.optimize_for_inference(model_jit)

    torch.jit.save(model_jit, output)


def export_pt2_bf16(model, *, inp_shapes, device, output):
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
    inp1 = torch.randn(1, *inp_shapes[0], dtype=torch.float32).to(device=device)
    inp2 = torch.randn(1, *inp_shapes[1], dtype=torch.float32).to(device=device)
    batch = torch.export.Dim("batch")

    with torch.no_grad():
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, cache_enabled=False
        ):
            ep = torch.export.export(
                model, args=(inp1, inp2), dynamic_shapes=({0: batch},), strict=True
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


def export_onnx(model, *, inp_shapes, device, output, fp16):
    inp1 = torch.randn(1, *inp_shapes[0], dtype=torch.float32).to(device=device)
    inp2 = torch.randn(1, *inp_shapes[1], dtype=torch.float32).to(device=device)

    with unlinking(tempfile.mkstemp(suffix=".onnx")) as tmp:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (inp1, inp2),
                tmp,
                input_names=["boards", "meta"],
                output_names=["policy", "value"],
            )

        if not fp16:
            import shutil

            shutil.copyfile(tmp, output)
            return

        import onnx
        from onnxconverter_common import auto_convert_mixed_precision

        model = onnx.load(tmp)
        model = auto_convert_mixed_precision(
            model,
            {"boards": inp1.cpu().numpy(), "meta": inp2.cpu().numpy()},
            rtol=0.01,
            atol=0.001,
            keep_io_types=True,
        )
        onnx.save(model, output)


def export_onnx2(model, *, inp_shapes, device, output, fp16):
    inp1 = torch.randn(1, *inp_shapes[0], dtype=torch.float32).to(device=device)
    inp2 = torch.randn(1, *inp_shapes[1], dtype=torch.float32).to(device=device)

    with torch.autocast(enabled=fp16, device_type=device):
        torch.onnx.export(
            model,
            (inp1, inp2),
            output,
            opset_version=18,
            export_params=True,
            do_constant_folding=True,
            dynamo=True,
            input_names=["boards", "meta"],
            output_names=["policy", "value"],
            dynamic_axes={
                "boards": {0: "batch_size"},
                "meta": {0: "batch_size"},
            },
        )
