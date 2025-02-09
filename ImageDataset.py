from PIL import Image
import os
from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, root_path, label_df, transform=None):
        self.images = [
            os.path.join(root_path, img_name) for img_name in label_df["file_name"]
        ]
        self.labels = label_df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

class InferenceImageDataset(Dataset):
    def __init__(self, root_path, label_df, transform=None):
        self.images = [
            os.path.join(root_path, img_name) for img_name in label_df["id"]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image