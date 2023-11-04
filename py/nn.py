import torch

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = relu(out)
        return out


class ChessModule(torch.nn.Module):
    N_RES_BLOCKS = 19

    def __init__(self):
        super().__init__()

        # 8 boards (14 channels each) + meta (7 channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(14 * 8 + 7, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(self.N_RES_BLOCKS)] )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8*8*128, 8*8*73),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x)
        v2 = self.value_head(x)
        return v1, v2
