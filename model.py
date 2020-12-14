from torch import nn
from torch import cat


# base conv block
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


# base conv transpose block
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


# base fully connected block
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


# plain encoder
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True)
        self.conv_3 = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True)
        self.norm_1 = nn.BatchNorm2d(num_features=in_ch)
        self.norm_2 = nn.BatchNorm2d(num_features=in_ch)
        self.norm_3 = nn.BatchNorm2d(num_features=out_ch)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.lrelu(x)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.lrelu(x)

        x = self.conv_3(x)
        x = self.norm_3(x)
        x = self.lrelu(x)

        return x


# plain decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1, bias=True)
        self.deconv_2 = nn.ConvTranspose2d(in_ch, in_ch, 3, 1, 1, bias=True)
        self.deconv_3 = nn.ConvTranspose2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.norm_1 = nn.BatchNorm2d(num_features=in_ch)
        self.norm_2 = nn.BatchNorm2d(num_features=in_ch)
        self.norm_3 = nn.BatchNorm2d(num_features=out_ch)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.deconv_1(x)
        x = self.norm_1(x)
        x = self.lrelu(x)

        x = self.deconv_2(x)
        x = self.norm_2(x)
        x = self.lrelu(x)

        x = self.deconv_3(x)
        x = self.norm_3(x)
        x = self.lrelu(x)

        return x


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

        self.top = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)

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


# plain convolution auto encoder
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.input_img_size = 64
        self.ae_layer_num = 4
        self.mlp_layer_num = 1
        self.inc = 1
        self.first_outc = 64

        self.input_img_element = self.input_img_size ** 2
        self.hidden_dim = self.input_img_size // 2 ** self.ae_layer_num

        self.encoder = self.make_encoder_layer()
        self.mlp = self.make_mlp_layer()
        self.decoder = self.make_decoder_layer()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def make_encoder_layer(self):
        layers = []
        inc = self.inc
        outc = self.first_outc

        for i in range(self.ae_layer_num):
            if i == 0:
                layers.append(EncoderBlock(inc, outc))
            else:
                layers.append(EncoderBlock(inc, outc))

            inc = outc
            outc = outc * 2

        self.inc = inc
        return nn.Sequential(*layers)

    def make_decoder_layer(self):
        layers = []
        inc = self.inc
        outc = self.inc // 2

        for i in range(self.ae_layer_num-1, -1, -1):
            if i == 0:
                layers.append(DecoderBlock(inc, 1))
            else:
                layers.append(DecoderBlock(inc, outc))

            inc = outc
            outc = inc // 2

        return nn.Sequential(*layers)

    def make_mlp_layer(self):
        layers = []
        dim = self.hidden_dim * self.inc

        for i in range(self.mlp_layer_num):
            layers.append(MlpBlock(dim, dim))

        return nn.Sequential(*layers)

# plain discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.img_ch = 1
        self.fc_ch = 512

        self.represent_cfg = [32, 32, 'P', 64, 64, 'P',
                              128, 128, 'P', 128, 128, 'P',
                              256, 256, 'P', 512, self.fc_ch, 'P']
        self.fc_cfg = [self.fc_ch, 256, 1]
        self.represent = self.make_represent_layers()
        self.fc = self.make_fc_layer()

    def forward(self, x):
        x = self.represent(x)
        x = x.view(-1, self.fc_ch)
        x = self.fc(x)

        return x

    def make_represent_layers(self):
        layers = []
        img_ch = self.img_ch

        for l in self.represent_cfg:
            if l == 'P':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(img_ch, l, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(l))
                layers.append(nn.LeakyReLU(0.2))

                img_ch = l

        return nn.Sequential(*layers)

    def make_fc_layer(self):
        layers = []
        fc_ch = self.fc_ch

        for l in self.fc_cfg:
            layers.append(nn.Linear(fc_ch, l))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))

            fc_ch = l

        return nn.Sequential(*layers)


if __name__ == '__main__':
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RU_AE().to(device)

    x = torch.randn((2, 1, 64, 64)).to(device)

    out = model(x)
    print(out.shape)