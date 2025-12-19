import torch
from torch._prims_common import check
from torchvision.ops import SqueezeExcitation
from timm.layers.norm import LayerNorm2d

NormTable = {
    "BatchNorm": torch.nn.BatchNorm2d,
    "LayerNorm": LayerNorm2d,
}

# import torch_tensorrt


class ResBlockSE(torch.nn.Module):
    def __init__(
        self, inplanes=256, planes=256, stride=1, use_se=True, norm="LayerNorm"
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        norm_builder = NormTable[norm]
        self.bn1 = norm_builder(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_builder(planes)
        self.se = (
            SqueezeExcitation(
                planes,
                planes // 2,
            )
            if use_se
            else torch.nn.Identity()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + x
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


class PolicyHead(torch.nn.Module):
    def __init__(self, din, norm):
        super().__init__()
        norm_builder = NormTable[norm]
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(din, 256, kernel_size=1, bias=False),
            norm_builder(256),
            torch.nn.Conv2d(din, 73, kernel_size=1, bias=False),
            norm_builder(73),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        x = self.model(x)
        return x.max(), torch.nn.functional.log_softmax(x, dim=1)


class ValueHead(torch.nn.Module):
    def __init__(self, din, norm):
        super().__init__()
        nf = 256
        norm_builder = NormTable[norm]
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(din, nf, kernel_size=1, bias=False),
            norm_builder(nf),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * nf + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x, meta):
        x = self.conv(x)
        # meta = meta.to(x.dtype)
        x = torch.cat((x, meta), dim=1)
        return self.ffn(x)


class ChessModule(torch.nn.Module):
    def __init__(self, n_res_blocks=19, n_boards=8, use_se=True, norm="LayerNorm"):
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
            NormTable[norm](256),
            torch.nn.ReLU(inplace=False),
        )

        self.res_blocks = torch.nn.ModuleList(
            [ResBlockSE(use_se=use_se, norm=norm) for _ in range(n_res_blocks)]
        )

        self.value_head = ValueHead(256, norm=norm)
        self.policy_head = PolicyHead(256, norm=norm)

    def forward(self, inp, meta):
        # inp shape bx112x8x8
        # meta shape bx7
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        latent = x

        max_logit, v1 = self.policy_head(latent)

        v2 = self.value_head(latent, meta)
        turn = meta[:, 0].unsqueeze(-1)
        v2 = v2 * (turn * 2 - 1)

        if self.training:
            return (v1, v2, max_logit)

        return (v1, v2)


def _load_ckpt(model, checkpoint, omit=None):
    print("..loading checkpoint: ", checkpoint)
    ckpt = torch.load(checkpoint, weights_only=True)

    if omit is not None:
        del_keys = [
            key for key in ckpt.keys() if any(key.startswith(pfx) for pfx in omit)
        ]
        for key in del_keys:
            del ckpt[key]

    if "pytorch-lightning_version" not in ckpt:
        state_dict = ckpt
    else:

        def _chop_prefix(name):
            return name.split(".", maxsplit=1)[1]

        state_dict = {_chop_prefix(k): v for k, v in ckpt["state_dict"].items()}

    r = model.load_state_dict(state_dict, strict=omit is None)
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


def _wrapup_py(model):

    def inference(*inp):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
            ):
                return model(*inp)

    return inference


def load_model_for_inference(checkpoint, n_res_blocks=None):
    if checkpoint.endswith(".ckpt"):
        assert n_res_blocks is not None
        return _wrapup_py(
            load_model(
                n_res_blocks=int(n_res_blocks),
                checkpoint=checkpoint,
                inference=True,
                device="cuda",
            )
        )

    if checkpoint.endswith(".pt"):
        return torch.jit.load(checkpoint)

    if checkpoint.endswith(".pt2"):
        return torch._inductor.aoti_load_package(checkpoint)
        # return _wrapup_py(torch.export.load(checkpoint).module())

    if checkpoint.endswith(".onnx"):
        import onnxruntime as ort

        sess = ort.InferenceSession(checkpoint)

        def inference(inp):
            p, v = sess.run(["policy", "value"], {"inp": inp.float().cpu().numpy()})
            return torch.from_numpy(p), torch.from_numpy(v)

        return inference

    raise RuntimeError("unsupported model " + checkpoint)
