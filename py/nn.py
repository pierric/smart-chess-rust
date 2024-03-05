import torch

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


class ChessModule(torch.nn.Module):
    N_RES_BLOCKS = 6

    def __init__(self):
        super().__init__()

        # 8 boards (14 channels each) + meta (7 channels)
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(14 * 8 + 7, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList([ResBlock() for _ in range(self.N_RES_BLOCKS)] )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=False),
            torch.nn.Flatten(),
            torch.nn.Linear(8*8*128, 8*8*73),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v1 = torch.log_softmax(v1, dim=1)

        v2 = self.value_head(x)
        return v1, v2


def _load_ckpt(model, checkpoint):
    print("..loading checkpoint: ", checkpoint)
    r = model.load_state_dict(torch.load(checkpoint), strict=True)
    if r.missing_keys:
        print("missing keys", r.missing_keys)
    if r.unexpected_keys:
        print("unexpected keys", r.unexpected_keys)


def load_model(device=None, checkpoint=None, inference=True, compile=True):
    """
    compile: not possible for training with quantized model
    device: can be true for training no matter if quantized or not
    """
    torch.manual_seed(0)
    
    device = device or "cpu"

    model = ChessModule()

    _load_ckpt(model, checkpoint)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    
    return model.to(device)


def export_ptq(checkpoint, output, *, calib):
    import pytorch_quantization
    import pytorch_quantization.quant_modules
    from itertools import islice
    from dataset import ChessDataset
    from torch.utils.data import ConcatDataset, DataLoader

    pytorch_quantization.quant_modules.initialize()

    model = ChessModule()
    _load_ckpt(model, checkpoint)

    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.enable_calib()
            module.disable_quant()

    dss = ConcatDataset([ChessDataset(trace_file) for trace_file in calib])
    train_loader = DataLoader(dss, num_workers=4, batch_size=1, shuffle=True, drop_last=True)
    for example in islice(train_loader, 50):
        model(example[0])

    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.load_calib_amax()
            module.disable_calib()
            module.enable_quant()

    model = model.cuda()

    dummy_input = torch.randn(1, 119, 8, 8, dtype=torch.float32, device='cuda')

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


def export_fp16(checkpoint, output):
    # assuming the model was trained with AMP
    # TODO I have to mask the quantized logic in the ChessModule too
    model = ChessModule()
    _load_ckpt(model, checkpoint)
    model.cuda().eval()

    x = torch.randn(1, 119, 8, 8, dtype=torch.float32).cuda()
    
    with torch.no_grad():
        # cache_enabled is critical to "trace" to the model
        # "script" works fine only for the non-amp model
        with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
            model_jit = torch.jit.trace(model, [x])
            model_jit = torch.jit.freeze(model_jit)

    import torch_tensorrt
    compiled = torch_tensorrt.compile(
        model_jit,
        inputs=[torch_tensorrt.Input((1, 119, 8, 8))],
        enabled_precisions=[torch.float, torch.half],
        ir="torchscript",
    )

    torch.jit.save(compiled, output)