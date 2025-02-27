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


class ChessModule(torch.nn.Module):

    def __init__(self, n_res_blocks=19):
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
            [ResBlock() for _ in range(n_res_blocks)]
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 16, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
            # torch.nn.Conv2d(256, 64, kernel_size=1, bias=False),
            # torch.nn.BatchNorm2d(64),
            # # # torch.nn.AvgPool2d(kernel_size=8),
            # torch.nn.Dropout2d(p=0.5),
            # torch.nn.Flatten(),
            # torch.nn.Linear(64 * 8 * 8, 512),
            # torch.nn.ReLU(inplace=False),
            # torch.nn.Linear(512, 1),
            # torch.nn.Tanh(),
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

    model = ChessModule(n_res_blocks=n_res_blocks)

    if checkpoint:
        _load_ckpt(model, checkpoint)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model.to(device)
