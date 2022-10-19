#! python
import csv
import json
from os import path

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.hifigan_dataset import HifiGanDataset
from model.hifi_gan import HifiGan
from util.args import args

if __name__ == "__main__":
    with open(args.config, "r") as infile:
        config = json.load(infile)

    if args.mode == "train":
        wav_dir = args.wav_dir

        train_data = pd.read_csv(
            args.training_data, delimiter="|", quoting=csv.QUOTE_NONE
        )
        train_dataset = HifiGanDataset(wav_dir, train_data.wav.tolist())
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
        val_dataset = HifiGanDataset(wav_dir, train_data.wav.tolist())
        val_dataloader = DataLoader(
            train_dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **config["dataloader"]
        )

        hifi_gan = HifiGan()
        hifi_gan.weight_norm()

        trainer = pl.Trainer(devices=[args.device], accelerator='gpu', precision=16,**config['trainer'])

        trainer.fit(hifi_gan, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    elif args.mode == "test":
        print("Testing")
    elif args.mode == "torchscript":
        print("Exporting torchscript")
