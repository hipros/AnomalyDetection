import torch
import argparse
from tqdm import tqdm

from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model import CAE
from torch import nn
from util import DatasetFromFolder, tensor_to_image
from torch.utils.data import DataLoader
from torch.autograd import Variable


class Solver(object):
    def __init__(self, config):
        self.lr = config.lr
        self.train_batch_size = config.trainBatch
        self.valid_batch_size = config.validBatch
        self.weight_decay = config.weightDecay
        self.device = config.cuda
        self.epoch = config.epoch
        self.image_resize = config.image_size
        self.saved_model = config.saved_model
        self.image_save = 1

        self.criterion = None
        self.optimizer = None
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.dtype = None

    def load_data(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])

        train_dataset = DatasetFromFolder('data/image/train', transform=train_transform)
        valid_dataset = DatasetFromFolder('data/image/valid', transform=valid_transform)

        self.train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=self.train_batch_size,
                                       shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, num_workers=8, batch_size=self.valid_batch_size,
                                       shuffle=False, drop_last=True)

        self.dtype = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor

    def load_model(self):
        self.model = CAE().to(self.device)

        if self.saved_model is not None:
            self.model.load_state_dict(torch.load(self.saved_model, map_location=self.device))

        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self):
        print("train: ")
        train_loss = 0
        self.model.train()

        for _, img in enumerate(tqdm(self.train_loader)):
            img = img.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(img)
            loss = self.criterion(output, img)

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        print("total loss = ", train_loss)

    def valid(self, epoch):
        print("valid: ")
        valid_loss = 0
        self.model.eval()

        for _, img in enumerate(tqdm(self.valid_loader)):
            img = img.to(self.device)

            output = self.model(img)
            loss = self.criterion(output, img)
            valid_loss += loss.item()

        print("total loss = ", valid_loss)
        if epoch % self.image_save == 0:
            torch.save(self.model.state_dict(), 'model_{}.pth'.format(epoch))
            save_image(make_grid(tensor_to_image(output, self.image_resize), nrow=8),
                       'data/result/image_{}.png'.format(epoch))

    def run(self):
        self.load_data()
        self.load_model()
        print(self.model)

        for epoch in range(1, self.epoch+1):
            print("\n===> epoch: {}/{}".format(epoch, self.epoch+1))
            self.train()
            self.valid(epoch)


def main():
    parser = argparse.ArgumentParser(description="NewPaerImplement")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--trainBatch', default=128, type=int)
    parser.add_argument('--validBatch', default=128, type=int)
    parser.add_argument('--weightDecay', default=0.001, type=float)
    parser.add_argument('--cuda', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--saved_model', default=None, type=str)
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
