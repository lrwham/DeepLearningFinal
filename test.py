if __name__ == "__main__":
    import lightning as L
    import torch
    import torch.nn as nn
    import os
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import pandas as pd
    from datamodule import DataModule
    
    # suggested by the debug output, set precission medium or high
    # read more at https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    torch.set_float32_matmul_precision("high")

    from model import CNN_Binary_Classifier, ResNetBinaryClassifier

    # load from checkpoint
    model = "checkpoints/2025-02-09-17-44-epoch=09-val_loss=0.13.ckpt"
    model = CNN_Binary_Classifier.load_from_checkpoint(model)
    
    # model = torch.compile(model)
    # check out https://lightning.ai/docs/pytorch/stable/advanced/speed.html#speed-amp
    # for some of these details
    trainer = L.Trainer(
        accelerator="gpu",
        devices="1",
        precision="16-mixed",
    )

    # run test dataset
    path = "/teamspace/studios/this_studio"
    print("About to test! 🎸")
    trainer.test(model, datamodule=DataModule(path))
