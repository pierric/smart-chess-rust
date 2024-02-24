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
        out += residual
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
            torch.nn.LeakyReLU(inplace=True),
        )

        self.res_blocks = torch.nn.ModuleList([ResBlock() for _ in range(self.N_RES_BLOCKS)] )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(8*8*128, 8*8*73),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v2 = self.value_head(x)
        return v1, v2


def _load_ckpt(model, checkpoint):
    print("..loading checkpoint: ", checkpoint)
    r = model.load_state_dict(torch.load(checkpoint), strict=True)
    if r.missing_keys:
        print("missing keys", r.missing_keys)
    if r.unexpected_keys:
        print("unexpected keys", r.unexpected_keys)


def load_model(device=None, checkpoint=None, inference=True):
    torch.manual_seed(0)
    
    device = device or "cpu"
    model = ChessModule()

    if checkpoint:
        _load_ckpt(model, checkpoint)

    if inference:
        model.eval()

    return torch.compile(model, mode="reduce-overhead", fullgraph=True).to(device)


def export(checkpoint, output):
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

