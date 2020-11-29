from torch import nn
import math


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        chpivet = 2
        mlp_depth = 1
        self.hidden_dim = 512

        encoder_module = []
        decoder_module = []
        mlp_module = []

        encoder_module.append(nn.Conv2d(1, chpivet, 4, 2, 1))
        encoder_module.append(nn.ReLU(True))
        encoder_module.append(nn.BatchNorm2d(chpivet))

        for i in range(int(math.log2(self.hidden_dim) - math.log2(chpivet))):
            inc = chpivet*(2**i); outc = chpivet*(2**(i+1))
            encoder_module.append(nn.Conv2d(inc, outc, 4, 2, 1))
            encoder_module.append(nn.ReLU(True))
            encoder_module.append(nn.BatchNorm2d(outc))

        for i in range(int(math.log2(self.hidden_dim) - math.log2(chpivet))-1, -1, -1):
            inc = chpivet*(2**(i+1)); outc = chpivet*(2**i)
            decoder_module.append(nn.ConvTranspose2d(inc, outc, 4, 2, 1))
            decoder_module.append(nn.ReLU(True))
            decoder_module.append(nn.BatchNorm2d(outc))

        decoder_module.append(nn.ConvTranspose2d(chpivet, 1, 4, 2, 1))

        for i in range(mlp_depth):
            mlp_module.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            mlp_module.append(nn.ReLU(True))
            mlp_module.append(nn.BatchNorm1d(self.hidden_dim))

        self.encoder = nn.Sequential(*encoder_module)
        self.mlp = nn.Sequential(*mlp_module)
        self.decoder = nn.Sequential(*decoder_module)

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(-1, self.hidden_dim)
        x = self.mlp(x)
        x = x.view(-1, self.hidden_dim, 1, 1)

        x = self.decoder(x)

        return x