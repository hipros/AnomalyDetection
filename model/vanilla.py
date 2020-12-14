from torch import nn
from .layer import FBR


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
            layers.append(FBR(dim, dim))

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

    model = CAE().to(device)

    x = torch.randn((2, 1, 64, 64)).to(device)

    out = model(x)
    print(out.shape)