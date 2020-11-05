from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from PIL import Image
import random
random.seed(5242)


class MedicalImageDataset(Dataset):

    def __init__(self, image_dir, label_path=None, transform=None, test=False):
        self.image_dir = image_dir
        self.test = test
        if not self.test:
            self.label_data = pd.read_csv(label_path)
            self.size = len(self.label_data["ID"].tolist())
            self.labels = self.label_data.iloc[:, 1].values
            self.index = self.label_data.iloc[:, 0].values
        else:
            self.label_data = None
            self.size = 292     # hard code this
        self.transform = transform

    def __len__(self):
        return self.size

    def _load_img(self, item):
        img_idx = item if self.test else self.label_data.iloc[item]["ID"]
        img_path = os.path.join(self.image_dir, "{}.png".format(img_idx))
        image = Image.open(img_path)
        return image, img_idx

    def __getitem__(self, anchor_item):
        if torch.is_tensor(anchor_item):
            anchor_item = anchor_item.tolist()

        anchor_img, anchor_idx = self._load_img(anchor_item)

        if self.transform:
            anchor_img = self.transform(anchor_img)

        anchor_label = -1 if self.test else self.label_data.iloc[anchor_item]["Label"]
        sample = {
            "id": anchor_idx,
            "image": anchor_img,
            "label": anchor_label
        }

        if not self.test:   # load triplet loss
            pos_list = self.index[self.index != anchor_item][self.labels[self.index != anchor_item] == anchor_label]
            pos_item = random.choice(pos_list)
            pos_img = Image.open(os.path.join(self.image_dir, "{}.png".format(pos_item)))

            neg_list = self.index[self.index != anchor_item][self.labels[self.index != anchor_item] != anchor_label]
            neg_item = random.choice(neg_list)
            neg_img = Image.open(os.path.join(self.image_dir, "{}.png".format(neg_item)))

            if self.transform:
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)

                sample["pos_image"] = pos_img
                sample["neg_image"] = neg_img

        return sample
