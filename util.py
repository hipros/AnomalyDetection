import os
from glob import glob

from torch.utils.data.dataset import Dataset
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

