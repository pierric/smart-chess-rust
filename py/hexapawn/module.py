import torch

class HexapawnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                2 * 4 + 3, 32, kernel_size=3, stride=1, padding=1, bias=False
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
