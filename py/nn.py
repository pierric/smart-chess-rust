import torch
# import torch_tensorrt


class ResBlock(torch.nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = torch.relu(out)
        return out


class ChessModuleBase(torch.nn.Module):
    N_RES_BLOCKS: int

    def __init__(self):
        super().__init__()

        # 8 boards (14 channels each) + meta (7 channels)
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                14 * 8 + 7, 256, kernel_size=5, stride=1, padding=2, bias=False
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList(
            [ResBlock() for _ in range(self.N_RES_BLOCKS)]
        )

        self.value_head = torch.nn.Sequential(
            # torch.nn.Conv2d(256, 1, kernel_size=1, bias=False),
            # torch.nn.BatchNorm2d(1),
            # torch.nn.Flatten(),
            # torch.nn.Linear(64, 64),
            # torch.nn.ReLU(inplace=False),
            # torch.nn.Linear(64, 1),
            # torch.nn.Tanh(),
            torch.nn.Conv2d(256, 64, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(64),
            # torch.nn.Dropout2d(p=0.5),
            torch.nn.AvgPool2d(kernel_size=8),
            torch.nn.Flatten(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(64, 1),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8 * 8 * 128, 8 * 8 * 73),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v1 = torch.log_softmax(v1, dim=1)

        v2 = self.value_head(x)
        return v1, v2


class ChessModule6(ChessModuleBase):
    N_RES_BLOCKS = 6


class ChessModule10(ChessModuleBase):
    N_RES_BLOCKS = 10


class ChessModule19(ChessModuleBase):
    N_RES_BLOCKS = 19


def chess_module_with(*, n_res_blocks):
    class DynChessModule(ChessModuleBase):
        N_RES_BLOCKS = n_res_blocks

    return DynChessModule()


def _load_ckpt(model, checkpoint):
    print("..loading checkpoint: ", checkpoint)
    r = model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    if r.missing_keys:
        print("missing keys", r.missing_keys)
    if r.unexpected_keys:
        print("unexpected keys", r.unexpected_keys)


def load_model(
    *, n_res_blocks=19, device=None, checkpoint=None, inference=True, compile=True
):
    """
    compile: not possible for training with quantized model
    device: can be true for training no matter if quantized or not
    """
    torch.manual_seed(0)

    device = device or "cpu"

    model = chess_module_with(n_res_blocks=n_res_blocks)

    if checkpoint:
        _load_ckpt(model, checkpoint)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model.to(device)


def export_ptq(n_res_blocks, *, checkpoint, output, calib):
    import pytorch_quantization
    import pytorch_quantization.quant_modules
    from itertools import islice
    from dataset import ChessDataset
    from torch.utils.data import ConcatDataset, DataLoader

    pytorch_quantization.quant_modules.initialize()

    model = chess_module_with(n_res_blocks=n_res_blocks)
    _load_ckpt(model, checkpoint)

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

    with pytorch_quantization.enable_onnx_export():
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


def export_fp16(n_res_blocks=19, *, checkpoint=None, output):
    # assuming the model was trained with AMP
    model = load_model(
        n_res_blocks=n_res_blocks,
        device="cuda",
        checkpoint=checkpoint,
        inference=True,
        compile=False,
    )

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


def export_bf16(n_res_blocks=19, *, checkpoint=None, output):
    # assuming the model was trained with AMP
    model = load_model(
        n_res_blocks=n_res_blocks,
        device="cuda",
        checkpoint=checkpoint,
        inference=True,
        compile=False,
    )
    # model.bfloat16()
    # model_jit = torch.jit.script(model)
    # model_jit = torch.jit.optimize_for_inference(model_jit)

    x = torch.randn(1, 119, 8, 8, dtype=torch.bfloat16).cuda()

    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            model_jit = torch.jit.trace(model, [x])
            model_jit = torch.jit.optimize_for_inference(model_jit)

    torch.jit.save(model_jit, output)


def export_pt2_bf16(n_res_blocks=19, *, checkpoint=None, output):
    torch.set_float32_matmul_precision("high")
    # assuming the model was trained with AMP
    model = load_model(
        n_res_blocks=n_res_blocks,
        device="cuda",
        checkpoint=checkpoint,
        inference=True,
        compile=False,
    )

    x = torch.randn(2, 119, 8, 8, dtype=torch.bfloat16).cuda()

    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            # model_jit = torch.jit.trace(model, [x])
            # model_jit = torch.jit.optimize_for_inference(model_jit)
            # torch.jit.save(model_jit, output)

            batch_dim = torch.export.Dim("batch", min=1, max=1024)
            ep = torch.export.export(
                model, (x,), dynamic_shapes={"inp": {0: batch_dim}}
            )
            torch._inductor.aoti_compile_and_package(ep, package_path=output)
