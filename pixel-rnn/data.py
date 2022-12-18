import torch
from torchvision import datasets, transforms

import pytorch_lightning as pl

BASE_DATA_DIR = "./data"

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_transform: transforms,
            data_dir: str = BASE_DATA_DIR,
            batch_size: int = 32,
            num_workers: int = 8,
        ):
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = transforms.ToTensor()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.train_transform
            )
            self.val_dset = datasets.CIFAR10(
                root=self.data_dir,
                rain=False,
                transform=self.val_transform
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )