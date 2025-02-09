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

   
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )

    from lightning.pytorch.callbacks import ModelCheckpoint
    import datetime

    # YYYY-MM-DD-HH-MM
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=timestamp + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )

    from model import CNN_Binary_Classifier, ResNetBinaryClassifier

    # model = CNN_Binary_Classifier(
    #     learning_rate=0.001
    # )
    model = CNN_Binary_Classifier(
        learning_rate=0.001,
    )
    
    # model = torch.compile(model)
    # check out https://lightning.ai/docs/pytorch/stable/advanced/speed.html#speed-amp
    # for some of these details
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision="16-mixed",
        max_time="00:04:00:00",
        max_epochs=10,
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
        profiler="simple",
    )

    path = "/teamspace/studios/this_studio"
    print("About to fit! 🎸")
    trainer.fit(model, datamodule=DataModule(path))