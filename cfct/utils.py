import os
import random
import numpy as np
import torch
from skimage import img_as_ubyte, io


class AvgMeter:
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(torch.tensor(val))

    def show(self):
        if len(self.losses) == 0:
            return 0.0
        return torch.mean(torch.stack(self.losses[max(len(self.losses) - self.num, 0):])).item()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_prediction_array(pred, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    io.imsave(os.path.join(output_dir, filename), img_as_ubyte(pred))
