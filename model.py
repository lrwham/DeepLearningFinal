import torch.nn as nn
import torch
import lightning as L
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class ResNetBinaryClassifier(L.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        self.resnet = resnet50() # weights=ResNet50_Weights.DEFAULT)
        old_fc = self.resnet.fc
        new_fc = nn.Sequential(
            nn.Linear(old_fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        self.resnet.fc = new_fc
        self.loss = nn.BCEWithLogitsLoss()
        self.learning_rate = 0.001 if learning_rate is None else learning_rate

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        # binary classification
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss


class CNN_Binary_Classifier(L.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        # inputs are 512x512 squares
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.loss = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.5)
        self.learning_rate = 0.001 if learning_rate is None else learning_rate

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.pool(
            self.leaky_relu(self.conv3(x))
        )  # Added forward pass for new layers
        x = self.pool(self.leaky_relu(self.conv4(x)))
        x = x.view(-1, 128 * 30 * 30)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        # binary classification
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1).float())
        self.log(
            "test_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
