import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CityScapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask