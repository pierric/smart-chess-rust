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

LOOKBACK = 2
META = 2

class HexapawnModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                2 * LOOKBACK + META, 32, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Flatten(),
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(3*3*32, 64),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(3*3 * 32, 3*3*6),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        v1 = self.policy_head(x)
        v1 = torch.log_softmax(v1, dim=1)

        v2 = self.value_head(x)
        return v1, v2


class HexapawnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                2 * LOOKBACK + META, 32, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList(
            [ResBlock(inplanes=32, planes=32) for _ in range(1)]
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*3*32, 64),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(3*3 * 32, 3*3*6),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v1 = torch.log_softmax(v1, dim=1)

        v2 = self.value_head(x)
        return v1, v2


def load_model(
    *, device=None, checkpoint=None, inference=True, compile=True
):
    torch.manual_seed(0)

    device = device or "cpu"

    model = HexapawnModule()

    if checkpoint:
        print("..loading checkpoint: ", checkpoint)
        r = model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        if r.missing_keys:
            print("missing keys", r.missing_keys)
        if r.unexpected_keys:
            print("unexpected keys", r.unexpected_keys)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model.to(device)
