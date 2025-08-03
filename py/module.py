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


class DropoutBlock(torch.nn.Module):
    def __init__(self, din: int, dout: int, p: float):
        super().__init__()
        self.model = torch.nn.Sequential(
            # torch.nn.Linear(din, dout),
            torch.nn.Conv2d(din, dout, kernel_size=1, bias=False, groups=din),
            torch.nn.BatchNorm2d(dout),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p),
        )

    def forward(self, x):
        return self.model(x)


class TwoLinearLayer(torch.nn.Module):
    def __init__(self, din: int, hidden: int, dout: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(din, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dout),
        )

    def forward(self, x):
        return self.model(x)


class PolicyHead(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(din, 73, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(73),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(8 * 8 * 73, 8 * 8 * 73),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class ValueHead(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(din, 4, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(4),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(8 * 8 * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class ChessModule(torch.nn.Module):
    def __init__(self, n_res_blocks=19, n_boards=8):
        super().__init__()

        # NOTE: The encoded boards (without the 7 meta planes) are passed through convoluational
        # layers. The resulting latent is then passed to the value and policy head. There
        # the 7 meta floats (meta planes are just a broadcast of these 7 floats to a 7x8x8 shape)
        # are concatenated to the latent before feeding into the Linear layers.

        # 8 boards (14 channels each)
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                14 * n_boards, 256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList([ResBlock() for _ in range(n_res_blocks)])

        # mid = 512
        # self.dropout = DropoutBlock(256, mid, 0.1)
        # self.value_head = TwoLinearLayer(8 * 8 * mid + 7, 512, 1)
        # self.policy_head = TwoLinearLayer(8 * 8 * mid + 7, 2048, 8 * 8 * 73)
        self.value_head = ValueHead(256)
        self.policy_head = PolicyHead(256)

    def forward(self, inp):
        # inp shape bx119x8x8
        # dim 0:112 are encoded boards
        # dim 112:119 are the meta data
        meta = inp[:, 112:, 0, 0]
        inp = inp[:, :112, ...]

        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)

        v2 = self.value_head(x)
        turn = meta[:, 0].unsqueeze(-1)
        v2 = v2 * (turn * 2 - 1)

        return v1, v2


def _load_ckpt(model, checkpoint, omit=None):
    print("..loading checkpoint: ", checkpoint)
    ckpt = torch.load(checkpoint, weights_only=True)

    if omit is not None:
        del_keys = [
            key for key in ckpt.keys() if any(key.startswith(pfx) for pfx in omit)
        ]
        for key in del_keys:
            del ckpt[key]

    r = model.load_state_dict(ckpt, strict=omit is None)
    if r.missing_keys:
        print("missing or omitted keys", r.missing_keys)
    if r.unexpected_keys:
        print("unexpected keys", r.unexpected_keys)


def load_model(
    *,
    n_res_blocks=19,
    device=None,
    checkpoint=None,
    inference=True,
    compile=True,
    omit=None,
):
    """
    compile: not possible for training with quantized model
    device: can be true for training no matter if quantized or not
    """
    torch.manual_seed(0)

    device = device or "cpu"

    model = ChessModule(n_res_blocks=n_res_blocks)

    if checkpoint:
        _load_ckpt(model, checkpoint, omit=omit)

    if inference:
        model.eval()

    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model.to(device)
