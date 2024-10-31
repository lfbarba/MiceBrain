import random

import h5py
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision import transforms

import torch.nn.functional as F
from PIL import Image
import math
import os
import numpy as np
from mice.datasets.base_dataset import BaseImageDataset, PairedTransform


Image.MAX_IMAGE_PIXELS = None

def apply_circle_mask(img):
    W, H = img.shape[-2:]
    cp = torch.cartesian_prod(torch.arange(W, device=img.device), torch.arange(H, device=img.device))
    circle_mask = (cp[:, 0] - W / 2) ** 2 + (cp[:, 1] - W / 2) ** 2 <= (W / 2) ** 2
    return img * circle_mask.reshape(img.shape[-2:])

class img_wrapper():
    def __init__(self, path, im_size=512):
        if os.path.isdir(path):
            folder = [filename for filename in os.listdir(path) if
                      filename.endswith('.npy')]
        else:
            folder = [os.path.basename(path)]
            path = os.path.dirname(path)

        self.path = path
        self.folder = folder
        self.im_size = im_size



    def __getitem__(self, idx):
        file_id = random.randint(0, len(self.folder)-1)
        filename = self.folder[file_id]
        image = torch.tensor(np.load(self.path + "/" + filename))
        w, h, c = image.shape
        coors = [
            random.randint(0, max(0, w - self.im_size - 1)),
            random.randint(0, max(0, h - self.im_size - 1))
        ]
        cropped_image = image[coors[0]:min(w, coors[0] + self.im_size), coors[1]:min(h, coors[1] + self.im_size)]
        return cropped_image.permute(2, 0, 1)

    def __len__(self):
        return 100000

class SliceDataset(BaseImageDataset):

    def __init__(self, path, im_size, lr_forward_function=lambda x:x,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None, num_defects=None,
                 contrast=None, train_transform=False, crop=None, gray_background=False,
                 to_synthetic=False, means=None, stds=None):
        super().__init__(path, lr_forward_function=lr_forward_function,
                 rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle, num_defects=num_defects,
                 contrast=contrast, train_transform=train_transform, crop=crop,
                 to_synthetic=to_synthetic)
        self.im_size = im_size
        self.images = img_wrapper(path, im_size)
        self.means = means
        self.stds = stds

        self.lr_forward_function = lr_forward_function

    def __getitem__(self, idx):
        hr_image = torch.tensor(np.array(self.images[idx])).float()

        if len(hr_image.shape) == 2:
            hr_image = hr_image.unsqueeze(0)

        hr_image = self.transform(hr_image)

        masked_img = apply_circle_mask(hr_image)
        stacked_img = torch.stack([masked_img[:, masked_img.shape[-1]//2], masked_img[:, :, masked_img.shape[-1]//2]])
        if self.means is not None:
            return (stacked_img - self.means[None, :, None]) / self.stds[None, :, None]
        else:
            return stacked_img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import lovely_tensors as lt
    import os
    lt.monkey_patch()

    DATA_PATH = '/mydata/chip/shared/data' if torch.cuda.is_available() else 'data'


    kwargs = {
        'path': 'data/sections',
        'im_size':20,
        'train_transform': True,
        'rotation_angle': 30,
        'rescale': 20,
    }

    trainSet = SliceDataset(**kwargs)
    print(trainSet[100])
