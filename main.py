import torch
import argparse
import os

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model import CAE
from model import Discriminator
from torch import nn
from util import DatasetFromFolder, tensor_to_image
from util import make_ones, make_zeros
from torch.utils.data import DataLoader


class Solver(object):
    def __init__(self, config):
        self.lr = config.lr
        self.train_batch_size = config.trainBatch
        self.valid_batch_size = config.validBatch
        self.weight_decay = config.weightDecay
        self.device = config.cuda
        self.epoch = config.epoch
        self.image_resize = config.imageSize
        self.saved_dir_model = config.savedDir
        self.train_dir = config.trainDir
        self.valid_dir = config.validDir
        self.image_save = True

        self.criterion_adv_gen = None
        self.criterion_adv_dis = None
        self.criterion_autoEn = None
        self.optimizer_adv_gen = None
        self.optimizer_adv_dis = None
        self.optimizer_autoEn = None
        self.model_gen = None
        self.model_dis = None
        self.train_loader = None
        self.valid_loader = None
        self.dtype = None

    def load_data(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_dataset = DatasetFromFolder(self.train_dir, transform=train_transform)
        valid_dataset = DatasetFromFolder(self.valid_dir, transform=valid_transform)

        self.train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=self.train_batch_size,
                                       shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, num_workers=8, batch_size=self.valid_batch_size,
                                       shuffle=False, drop_last=True)

        self.dtype = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor

    def load_model(self):
        self.model_gen = CAE().to(self.device)
        self.model_dis = Discriminator().to(self.device)

        if self.saved_dir_model is not None:
            saved_path_gen = self.saved_model_dir + 'gen.pth'
            saved_path_dis = self.saved_model_dir + 'dis.pth'

            self.model_gen.load_state_dict(torch.load(saved_path_gen, map_location=self.device))
            self.model_dis.load_state_dict(torch.load(saved_path_dis, map_location=self.device))

        self.criterion_adv_gen = nn.MSELoss().to(self.device)
        self.criterion_adv_dis = nn.MSELoss().to(self.device)
        self.criterion_autoEn = nn.MSELoss().to(self.device)

        self.optimizer_adv_gen = torch.optim.Adam(self.model_gen.parameters(), lr=self.lr,
                                                  weight_decay=self.weight_decay)
        self.optimizer_adv_dis = torch.optim.Adam(self.model_dis.parameters(), lr=self.lr,
                                                  weight_decay=self.weight_decay)
        self.optimizer_autoEn = torch.optim.Adam(self.model_gen.parameters(), lr=self.lr,
                                                 weight_decay=self.weight_decay)

    def train(self):
        print("train: ")
        train_loss = 0.0

        self.model_gen.train()
        self.model_dis.train()

        for _, img in enumerate(tqdm(self.train_loader)):
            img = img.to(self.device)
            img_ori = img

            # train autoEncoder
            img_gen = self.model_gen(img)
            loss_recon = self.train_autoencoder(img_gen, img_ori)

            # train discriminator
            img_gen = self.model_gen(img).detach()
            loss_adv_dis = self.train_discriminator(img_gen, img_ori)

            # train generator
            img_gen = self.model_gen(img)
            loss_adv_gen = self.train_generator(img_gen)

            train_loss += (loss_adv_gen.item() + loss_adv_dis.item() + loss_recon.item())

        print("total loss = ", train_loss)

    def valid(self, epoch):
        print("valid: ")
        valid_loss = 0.0
        img_gen = None

        self.model_dis.eval()
        self.model_gen.eval()

        for _, img in enumerate(tqdm(self.valid_loader)):
            n = img.size(0)
            img_ori = img.to(self.device)
            img_gen = self.model_gen(img_ori)

            output_dis_ori = self.model_dis(img_ori)
            output_dis_gen = self.model_dis(img_gen)

            loss_gen = self.criterion_adv_gen(img_gen, img_ori)
            loss_dis_ori = self.criterion_adv_dis(output_dis_ori, make_ones(n, self.device))
            loss_dis_gen = self.criterion_adv_dis(output_dis_gen, make_zeros(n, self.device))
            loss_recon = self.criterion_autoEn(img_ori, img_gen)

            valid_loss += loss_gen.item() + loss_dis_ori.item() + loss_dis_gen.item() + loss_recon.item()

        print('total valid loss = ', valid_loss)

        ## save results
        save_dir = os.path.join('saved_model', str(epoch))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.save_model(save_dir)
        self.save_image(save_dir, img_gen)

    def save_model(self, save_dir):
        save_path_gen = os.path.join(save_dir, 'gen.pth')
        torch.save(self.model_gen.state_dict(), save_path_gen)

        save_path_dis = os.path.join(save_dir, 'dis.pth')
        torch.save(self.model_dis.state_dict(), save_path_dis)

    def save_image(self, save_dir, img_gen):
        save_path_img = os.path.join(save_dir, 'gen_result.png')
        save_image(make_grid(tensor_to_image(img_gen, self.image_resize), nrow=8), save_path_img)

    def train_autoencoder(self, img_gen, img_ori):
        self.optimizer_autoEn.zero_grad()

        loss = self.criterion_autoEn(img_ori, img_gen)

        loss.backward()
        self.optimizer_autoEn.step()

        return loss

    def train_generator(self, img_gen):
        n = img_gen.size(0)

        self.optimizer_adv_gen.zero_grad()

        output_dis_gen = self.model_dis(img_gen)
        loss = self.criterion_adv_gen(output_dis_gen, make_ones(n, self.device))

        loss.backward()
        self.optimizer_adv_gen.step()

        return loss

    def train_discriminator(self, img_gen, ori_img):
        n = ori_img.size(0)

        self.optimizer_adv_dis.zero_grad()

        output_dis_ori = self.model_dis(ori_img)
        loss_ori = self.criterion_adv_dis(output_dis_ori, make_ones(n, self.device))
        loss_ori.backward()

        output_dis_gen = self.model_dis(img_gen)
        loss_gen = self.criterion_adv_dis(output_dis_gen, make_zeros(n, self.device))
        loss_gen.backward()

        self.optimizer_adv_dis.step()

        return loss_ori + loss_gen

    def run(self):
        self.load_data()
        self.load_model()
        print(self.model_gen)
        print(self.model_dis)

        for epoch in range(1, self.epoch + 1):
            print("\n===> epoch: {}/{}".format(epoch, self.epoch + 1))
            self.train()
            self.valid(epoch)


def main():
    parser = argparse.ArgumentParser(description="NewPaerImplement")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--trainBatch', default=128, type=int)
    parser.add_argument('--validBatch', default=128, type=int)
    parser.add_argument('--trainDir', default='data/image/train', type=str)
    parser.add_argument('--validDir', default='data/image/valid', type=str)
    parser.add_argument('--weightDecay', default=0.001, type=float)
    parser.add_argument('--cuda', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--imageSize', default=64, type=int)
    parser.add_argument('--savedDir', default=None, type=str)
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
