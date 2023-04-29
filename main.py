#! python
import csv
import json

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from hifi_gan.dataset import HifiGanDataset
from hifi_gan.model import HifiGan
from util.args import args

if __name__ == "__main__":
    with open(args.config, "r") as infile:
        config = json.load(infile)

    if args.mode == "train":
        wav_dir = args.wav_dir

        train_data = pd.read_csv(
            args.training_data, delimiter="|", quoting=csv.QUOTE_NONE
        )
        train_dataset = HifiGanDataset(
            wav_dir=wav_dir, files=train_data.wav.tolist(), **config["dataset"]
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **config["dataloader"]
        )

        val_data = pd.read_csv(
            args.validation_data, delimiter="|", quoting=csv.QUOTE_NONE
        )
        val_dataset = HifiGanDataset(wav_dir, val_data.wav.tolist())
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **config["dataloader"]
        )

        hifi_gan = HifiGan()

        trainer = pl.Trainer(
            devices=[args.device],
            accelerator="gpu",
            precision=16,
            **config["trainer"],
            callbacks=[LearningRateMonitor(logging_interval="epoch")]
        )

        trainer.fit(
            hifi_gan, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

    elif args.mode == "test":
        print("Testing")
    elif args.mode == "torchscript":
        print("Exporting torchscript")
