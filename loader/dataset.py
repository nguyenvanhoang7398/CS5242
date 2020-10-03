from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
from PIL import Image


class MedicalImageDataset(Dataset):

    def __init__(self, image_dir, label_path=None, transform=None, test=False):
        self.image_dir = image_dir
        self.test = test
        if not self.test:
            self.label_data = pd.read_csv(label_path)
            self.size = len(self.label_data["ID"].tolist())
        else:
            self.label_data = None
            self.size = 292     # hard code this
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_idx = item if self.test else self.label_data.iloc[item]["ID"]
        img_path = os.path.join(self.image_dir, "{}.png".format(img_idx))
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label = -1 if self.test else self.label_data.iloc[item]["Label"]
        sample = {
            "id": img_idx,
            "image": image,
            "label": label
        }

        return sample
