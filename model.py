from torch import nn


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


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True)
        self.deconv_2 = nn.ConvTranspose2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.deconv_3 = nn.ConvTranspose2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.norm_1 = nn.BatchNorm2d(num_features=out_ch)
        self.norm_2 = nn.BatchNorm2d(num_features=out_ch)
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


class MlpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Linear(in_ch, out_ch)
        self.norm = nn.BatchNorm1d(out_ch)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        x = self.lrelu(x)

        return x


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.input_img_size = 64
        self.ae_layer_num = 3
        self.mlp_layer_num = 1

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
        for i in range(self.ae_layer_num):
            inc = 2 ** i
            outc = 2 * inc

            layers.append(EncoderBlock(inc, outc))

        return nn.Sequential(*layers)

    def make_decoder_layer(self):
        layers = []

        for i in range(self.ae_layer_num-1, -1, -1):
            inc = 2 ** (i+1)
            outc = inc // 2

            layers.append(DecoderBlock(inc, outc))

        return nn.Sequential(*layers)

    def make_mlp_layer(self):
        layers = []

        for i in range(self.mlp_layer_num):
            layers.append(MlpBlock(self.hidden_dim, self.hidden_dim))

        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.img_ch = 1
        self.fc_ch = 1024

        self.represent_cfg = [64, 64, 'P', 128, 128, 'P',
                              256, 256, 256, 256, 'P',
                              512, 512, 512, 512, 'P',
                              1024, 1024, 'P', 1024, 1024, 'P']
        self.fc_cfg = [1024, 512, 256, 1]
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
