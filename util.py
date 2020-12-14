import os
import torch

from glob import glob
from torch.utils.data.dataset import Dataset
from torch.autograd.variable import Variable
from PIL import Image


class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, transform):
        super(DatasetFromFolder, self).__init__()

        self.image_dir = data_dir
        glob_str = os.path.join(self.image_dir, '*.png')
        self.image_files = [filename for filename in glob(glob_str)]
        self.transform = transform

    def __getitem__(self, ind):
        img, _, _ = Image.open(self.image_files[ind]).convert('YCbCr').split()

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_files)


def make_ones(size, device):
    return Variable(torch.ones(size, 1)).to(device)


def make_zeros(size, device):
    return Variable(torch.zeros(size, 1)).to(device)


def tensor_to_image(tensor_img, img_size):
    img = 0.5 * (tensor_img)
    img = img.clamp(0, 1)
    img = img.view(img.size(0), 1, img_size, img_size)

    return img

