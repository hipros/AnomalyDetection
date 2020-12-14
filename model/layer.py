from torch import nn


# base conv layer
class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, is_last=False):
        super(CBR, self).__init__()
        layer = []

        layer += [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias)]
        layer += [nn.BatchNorm2d(out_ch)]

        if not is_last:
            layer += [nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


# base conv transpose layer
class TBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, is_last=False):
        super(TBR, self).__init__()
        layer = []

        layer += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias)]
        layer += [nn.BatchNorm2d(out_ch)]

        if not is_last:
            layer += [nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


# base fully connected layer
class FBR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FBR, self).__init__()
        layers = []

        layers += [nn.Linear(in_ch, out_ch)]
        layers += [nn.BatchNorm1d(out_ch)]
        layers += [nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
