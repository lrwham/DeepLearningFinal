# load checkpoints/2025-02-06-23-45-epoch=10-val_loss=0.12.ckpt
import lightning as L
import torch
from model import CNN_Binary_Classifier, ResNetBinaryClassifier
from ImageDataset import InferenceImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from datamodule import DataModule
import datetime

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    path = "/teamspace/studios/this_studio"
    # load the model
    model = CNN_Binary_Classifier.load_from_checkpoint(
"checkpoints/2025-02-09-17-44-epoch=09-val_loss=0.13.ckpt"    )

    # use CPU
    trainer = L.Trainer(accelerator="gpu", devices=1)

    datamodule = DataModule(path)

    # validation = trainer.validate(model, datamodule=datamodule)

    # print(validation)

    predictions = trainer.predict(model, datamodule=datamodule)

    predictions = torch.cat(predictions).numpy().flatten()

    # round the predictions to 0 or 1
    predictions = (predictions > 0.4).astype(int)

    test_df = pd.read_csv(path + "/test.csv")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # save the predictions with their corresponding ids
    test_df["predictions"] = predictions
    test_df.to_csv(f"predictions-{now}.csv", index=False)
    print(f"Predictions saved to predictions-{now}.csv")




