from torch import nn


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.mlp = nn.Sequential()

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        x = self.decoder(x)
        return x