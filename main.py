#! python
import csv
import json

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
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
        val_dataset = HifiGanDataset(
            wav_dir=wav_dir, files=val_data.wav.tolist(), **config["dataset"]
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **config["dataloader"]
        )

        torch.set_float32_matmul_precision("high")

        hifi_gan = HifiGan()

        trainer = pl.Trainer(
            devices=[args.device],
            accelerator="gpu",
            precision="16-mixed",
            **config["trainer"],
            callbacks=[LearningRateMonitor(logging_interval="step")],
            benchmark=True
        )

        trainer.fit(
            hifi_gan,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint,
        )

    elif args.mode == "finetune":
        wav_dir = args.wav_dir
        finetune_dir = args.finetune_dir

        train_data = pd.read_csv(
            args.training_data, delimiter="|", quoting=csv.QUOTE_NONE
        )
        train_dataset = HifiGanDataset(
            wav_dir=wav_dir,
            files=train_data.wav.tolist(),
            finetune=True,
            finetune_dir=finetune_dir,
            **config["dataset"]
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
        val_dataset = HifiGanDataset(
            wav_dir, val_data.wav.tolist(), finetune=True, finetune_dir=finetune_dir
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **config["dataloader"]
        )

        torch.set_float32_matmul_precision("high")

        hifi_gan = HifiGan.load_from_checkpoint(args.finetune_ckpt)

        trainer = pl.Trainer(
            devices=[args.device],
            accelerator="gpu",
            precision="16-mixed",
            **config["trainer"],
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            benchmark=True
        )

        trainer.fit(
            hifi_gan,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint,
        )

    elif args.mode == "test":
        print("Testing")
    elif args.mode == "torchscript":
        print("Exporting torchscript")
