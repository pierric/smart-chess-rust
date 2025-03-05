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
    def __init__(self, n_res_blocks=19, n_boards=8):
        super().__init__()

        # NOTE: The encoded boards (without the 7 meta planes) are passed through convoluational
        # layers. The resulting latent is then passed to the value and policy head. There
        # the 7 meta floats (meta planes are just a broadcast of these 7 floats to a 7x8x8 shape)
        # are concatenated to the latent before feeding into the Linear layers.

        # 8 boards (14 channels each)
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                14 * n_boards, 256, kernel_size=5, stride=1, padding=2, bias=False
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList([ResBlock() for _ in range(n_res_blocks)])

        self.value_head_conv = torch.nn.Conv2d(256, 16, kernel_size=1, bias=False)
        self.value_head_bn = torch.nn.BatchNorm2d(16)
        # 7 floats of meta information are concatenated
        self.value_head_linear1 = torch.nn.Linear(8 * 8 * 16 + 7, 256)
        self.value_head_linear2 = torch.nn.Linear(256, 1)

        self.policy_head_conv = torch.nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.policy_head_bn = torch.nn.BatchNorm2d(128)
        self.policy_head_linear = torch.nn.Linear(8 * 8 * 128 + 7, 8 * 8 * 73)

    def value_head(self, x, meta):
        x = self.value_head_conv(x)
        x = self.value_head_bn(x)
        x = torch.flatten(x, 1, -1)
        x = torch.concatenate((x, meta), dim=1)
        x = self.value_head_linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.value_head_linear2(x)
        x = torch.nn.functional.tanh(x)

        turn = meta[:, 0].unsqueeze(-1)
        return x * (turn * 2 - 1)

    def policy_head(self, x, meta):
        x = self.policy_head_conv(x)
        x = self.policy_head_bn(x)
        x = torch.flatten(x, 1, -1)
        x = torch.concatenate((x, meta), dim=1)
        x = torch.nn.functional.relu(x)
        x = self.policy_head_linear(x)
        return torch.log_softmax(x, dim=1)

    def forward(self, inp):
        # inp shape bx119x8x8
        # dim 0:112 are encoded boards
        # dim 112:119 are the meta data
        meta = inp[:, 112:, 0, 0]
        inp = inp[:, :112, ...]

        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x, meta)
        v2 = self.value_head(x, meta)
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
