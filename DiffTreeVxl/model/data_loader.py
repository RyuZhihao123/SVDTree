import pytorch_lightning as pl
import numpy as np
import torch
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class TreeDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        img_size: int,
        debug,
    ):
        super(TreeDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_path = os.path.join(self.data_dir, "train")
        self.test_path = os.path.join(self.data_dir, "test")
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4

        train_dataset = occupancy_field_Dataset(self.train_path, img_size)
        self.train_data, self.val_data = random_split(train_dataset, [1100, 200])
        self.test_data = occupancy_field_Dataset(self.test_path, img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )


class occupancy_field_Dataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 img_size: int = 448,
    ):
        super().__init__()
        self.img_folder = os.path.join(data_folder, "img")
        self.vxl_folder = os.path.join(data_folder, "vxl")
        assert os.path.exists(self.img_folder), f"path '{self.img_folder}' does not exists."
        assert os.path.exists(self.vxl_folder), f"path '{self.vxl_folder}' does not exists."

        img_names = [i for i in os.listdir(self.img_folder) if i.endswith(".png")]
        self.img_list = [os.path.join(self.img_folder, i) for i in img_names]
        vxl_names = [i for i in os.listdir(self.vxl_folder) if i.endswith(".npy")]
        self.vxl_list = [os.path.join(self.vxl_folder, i) for i in vxl_names]

        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        img = Image.open(self.img_list[index])
        data["img"] = self.transform(img)
        voxel = np.load(self.vxl_list[index])
        data["voxel"] = voxel
        # print(data["img"].min(), data["img"].max())
        # transforms.ToPILImage()(data["img"]).show()
        # print(self.img_list[index], self.vxl_list[index])
        return data

# loader = occupancy_field_Dataset("../dataset/train")
# data = loader.__getitem__(0)
# print(data["voxel"].max(), data["voxel"].min())
# print(data["voxel"].shape)