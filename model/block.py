from torch import nn
from .layer import CBR, TBR, FBR


# down sampling block
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.down = CBR(in_ch, out_ch, 4, 2, 1)

    def forward(self, x):
        return self.down(x)


# Up sampling block
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = TBR(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


# residual encoder block
class REncoBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(REncoBlock, self).__init__()
        self.cbr_1 = CBR(in_ch, out_ch)
        self.cbr_2 = CBR(out_ch, out_ch, is_last=True)

        self.short = CBR(in_ch, out_ch, kernel_size=1, padding=0, is_last=True)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.cbr_1(x)
        out = self.cbr_2(out)

        x = self.short(x)

        out = self.lrelu(x+out)

        return out


# residual decoder block
class RDecoBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RDecoBlock, self).__init__()
        self.tbr_1 = TBR(in_ch, out_ch)
        self.tbr_2 = TBR(in_ch, out_ch, is_last=True)

        self.short = TBR(in_ch, out_ch, kernel_size=1, padding=0, is_last=True)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.tbr_1(x)
        out = self.tbr_2(out)

        x = self.short(x)

        out = self.lrelu(x+out)

        return out