from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True)
        self.norm_1 = nn.BatchNorm2d(num_features=in_ch)
        self.norm_2 = nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = F.leaky_relu(x, 0.2)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True)
        self.deconv_2 = nn.ConvTranspose2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.norm_1 = nn.BatchNorm2d(out_ch)
        self.norm_2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.deconv_1(x)
        x = self.norm_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.deconv_2(x)
        x = self.norm_2(x)
        x = F.leaky_relu(x, 0.2)

        return x


class MlpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Linear(in_ch, out_ch)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        x = F.leaky_relu(x, 0.2)

        return x


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.input_img_size = 64
        self.ae_layer_num = 3
        self.mlp_layer_num = 1

        self.input_img_element = self.input_img_size ** 2
        self.hidden_dim = self.input_img_size // 2 ** self.ae_layer_num

        encoder_module = []
        decoder_module = []
        mlp_module = []

        for i in range(self.ae_layer_num):
            inc = 2 ** i
            outc = 2 * inc

            encoder_module.append(EncoderBlock(inc, outc))

        for i in range(self.ae_layer_num-1, -1, -1):
            inc = 2 ** (i+1)
            outc = inc // 2

            decoder_module.append(DecoderBlock(inc, outc))

        for i in range(self.mlp_layer_num):
            mlp_module.append(MlpBlock(self.hidden_dim, self.hidden_dim))

        self.encoder = nn.Sequential(*encoder_module)
        self.mlp = nn.Sequential(*mlp_module)
        self.decoder = nn.Sequential(*decoder_module)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x