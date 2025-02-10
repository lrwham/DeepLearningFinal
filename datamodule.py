import lightning as L
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ImageDataset import ImageDataset, InferenceImageDataset


class DataModule(L.LightningDataModule):
    def __init__(self, path, batch_size=16, num_workers=11, fraction=1, persistent_workers=True):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fraction = fraction
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        train_df = pd.read_csv(self.path + "/train.csv")

        train_df = train_df.sample(frac=self.fraction, random_state=99)

        predict_df = pd.read_csv(self.path + "/test.csv")

        # From the ResNet documentation:
        # All pre-trained models expect input images normalized in the same way,
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        # where H and W are expected to be at least 224.
        #
        # The images have to be loaded in to a range of [0, 1]
        # and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

        # For this dataset at a 512x512 resolution, here are the mean and std values:
        mean = (0.6238067150115967, 0.5922583341598511, 0.5293540358543396)
        std = (0.3335755169391632, 0.3218654990196228, 0.3552871346473694)

        resnet_train_transform = transforms.Compose(
            [
                transforms.Resize(int(512 * 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        resnet_val_transform = transforms.Compose(
            [
                transforms.Resize(int(512 * 1.2)),
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_split_df, val_split_df = train_test_split(
            train_df, test_size=0.4, random_state=99
        )

        test_split_df, val_split_df = train_test_split(
            val_split_df, test_size=0.5, random_state=99
        )

        train_split_df = pd.DataFrame(train_split_df)
        val_split_df = pd.DataFrame(val_split_df)
        test_split_df = pd.DataFrame(test_split_df)

        train_split_df.reset_index(drop=True, inplace=True)
        val_split_df.reset_index(drop=True, inplace=True)
        test_split_df.reset_index(drop=True, inplace=True)

        self.train_dataset = ImageDataset(
            self.path, train_split_df, transform=resnet_train_transform
        )
        self.val_dataset = ImageDataset(
            self.path, val_split_df, transform=resnet_val_transform
        )
        self.test_dataset = ImageDataset(
            self.path, test_split_df, transform=resnet_val_transform
        )
        self.predict_dataset = InferenceImageDataset(
            self.path, predict_df, transform=resnet_val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=4,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=4,
            persistent_workers=self.persistent_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=4,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=4,
            persistent_workers=self.persistent_workers,
        )
