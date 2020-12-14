
from torch import nn
from torch import cat
from .block import REncoBlock, DownBlock, UpBlock


# residual U-net auto encoder
class RU_AE(nn.Module):
    def __init__(self):
        super(RU_AE, self).__init__()

        self.conv_1 = REncoBlock(1, 64)
        self.down_1 = DownBlock(64, 64)

        self.conv_2 = REncoBlock(64, 128)
        self.down_2 = DownBlock(128, 128)

        self.conv_3 = REncoBlock(128, 256)
        self.down_3 = DownBlock(256, 256)

        self.conv_4 = REncoBlock(256, 512)
        self.down_4 = DownBlock(512, 512)

        self.bottom = REncoBlock(512, 1024)

        self.up_4 = UpBlock(1024, 512)
        self.convt_4 = REncoBlock(1024, 512)

        self.up_3 = UpBlock(512, 256)
        self.convt_3 = REncoBlock(512, 256)

        self.up_2 = UpBlock(256, 128)
        self.convt_2 = REncoBlock(256, 128)

        self.up_1 = UpBlock(128, 64)
        self.convt_1 = REncoBlock(128, 64)

        self.top = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x = self.down_1(x_1)

        x_2 = self.conv_2(x)
        x = self.down_2(x_2)

        x_3 = self.conv_3(x)
        x = self.down_3(x_3)

        x_4 = self.conv_4(x)
        x = self.down_4(x_4)

        x = self.bottom(x)

        x = self.up_4(x)
        x = cat((x, x_4), dim=1)
        x = self.convt_4(x)

        x = self.up_3(x)
        x = cat((x, x_3), dim=1)
        x = self.convt_3(x)

        x = self.up_2(x)
        x = cat((x, x_2), dim=1)
        x = self.convt_2(x)

        x = self.up_1(x)
        x = cat((x, x_1), dim=1)
        x = self.convt_1(x)

        x = self.top(x)

        return x


if __name__ == '__main__':
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RU_AE().to(device)

    x = torch.randn((2, 1, 64, 64)).to(device)

    out = model(x)
    print(out.shape)