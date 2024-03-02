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
        self.func = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.func.add(out, residual)
        out = torch.relu(out)
        return out


class ChessModule(torch.nn.Module):
    N_RES_BLOCKS = 6

    def __init__(self):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant1 = torch.ao.quantization.DeQuantStub()
        self.dequant2 = torch.ao.quantization.DeQuantStub()

        # 8 boards (14 channels each) + meta (7 channels)
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(14 * 8 + 7, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList([ResBlock() for _ in range(self.N_RES_BLOCKS)] )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Flatten(),
            torch.nn.Linear(8*8*128, 8*8*73),
        )

    def forward(self, inp):
        x = self.quant(inp)

        x = self.conv_block(x)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v1 = self.dequant1(v1)
        v1 = torch.log_softmax(v1, dim=1)

        v2 = self.value_head(x)
        v2 = self.dequant2(v2)
        return v1, v2


def _load_ckpt(model, checkpoint):
    print("..loading checkpoint: ", checkpoint)
    r = model.load_state_dict(torch.load(checkpoint), strict=True)
    if r.missing_keys:
        print("missing keys", r.missing_keys)
    if r.unexpected_keys:
        print("unexpected keys", r.unexpected_keys)


def prepare_quantization(model):
    model.eval()

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    fuse_list = [
        ["conv_block.0", "conv_block.1"],
        ["value_head.0", "value_head.1"],
        ["policy_head.0", "policy_head.1"],
    ]
    for i in range(ChessModule.N_RES_BLOCKS):
        fuse_list.extend([
            [f"res_blocks.{i}.conv1", f"res_blocks.{i}.bn1"],
            [f"res_blocks.{i}.conv2", f"res_blocks.{i}.bn2"],
        ])
    
    model = torch.ao.quantization.fuse_modules(model, [
        ['res_blocks.5.conv1', 'res_blocks.5.bn1'],
        ['res_blocks.5.conv2', 'res_blocks.5.bn2']
    ])
    return torch.ao.quantization.prepare_qat(model.train())


def load_model(device=None, checkpoint=None, inference=True, compile=True, ckpt_quantized=False):
    """
    ckpt_quantized: false < runs/18, and true >= 18
    compile: not possible for training with quantized model
    device: can be true for training no matter if quantized or not
    """
    torch.manual_seed(0)
    
    device = device or "cpu"

    model = ChessModule()

    if checkpoint and not ckpt_quantized:
        _load_ckpt(model, checkpoint)

    model = prepare_quantization(model)

    if checkpoint and ckpt_quantized:
        _load_ckpt(model, checkpoint)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    
    return model.to(device)


def export(checkpoint, output):
    # assuming the model was trained in the QAT way
    model = ChessModule()
    model = prepare_quantization(model)
    _load_ckpt(model, checkpoint)
    model.eval()
    model = torch.ao.quantization.convert(model)

    x = torch.randn(1, 119, 8, 8, dtype=torch.float32)
    model_jit = torch.jit.trace(model, (x,))
    model_jit = torch.compile(model_jit)

    torch.jit.save(model_jit, output)


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

    torch.jit.save(model_jit, output)
    # re-load and save the model to cpu, as the C++ api cannot load for no good reason
    model_jit = torch.jit.load(output, map_location="cpu")
    torch.jit.save(model_jit, output)
