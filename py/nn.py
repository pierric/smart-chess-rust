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
    N_RES_BLOCKS = 19

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


def load_model():
    model = ChessModule()
    model.eval()
    return torch.compile(model, mode="reduce-overhead")
