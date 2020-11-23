import torch
import argparse

from torchvision import transforms
from model import CAE
from torch import nn
from util import DatasetFromFolder
from torch.utils.data import DataLoader


class Solver():
    def __init__(self, config):
        self.lr = config.lr
        self.train_batch_size = config.trainBatch
        self.valid_batch_size = config.validBatch
        self.weight_decay = config.weightDecay
        self.device = config.cuda
        self.epoch = config.epoch

        self.criterion = None
        self.optimizer = None
        self.model = None
        self.train_loader = None
        self.valid_loader = None

    def load_data(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = DatasetFromFolder('data/image/train', transform=train_transform)
        valid_dataset = DatasetFromFolder('data/image/valid', transform=valid_transform)

        self.train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=self.train_batch_size,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, num_workers=0, batch_size=self.valid_batch_size,
                                       shuffle=False)

    def load_model(self):
        self.model = CAE().to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)

    def train(self):
        pass

    def valid(self):
        pass

    def run(self):
        self.load_data()
        self.load_model()


def main():
    parser = argparse.ArgumentParser(description="NewPaerImplement")
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--trainBatch', default=32)
    parser.add_argument('--validBatch', default=32)
    parser.add_argument('--weightDecay', default=0.001)
    parser.add_argument('--cuda', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
