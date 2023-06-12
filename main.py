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
        train_data = train_data[train_data.duration >= 0.5]
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
        val_data = val_data[val_data.duration >= 0.5]
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

        if args.checkpoint and args.resume_checkpoint:
            raise Exception(
                "You may only specify a starting checkpoint (with --checkpoint) or a resume checkpoint (--resume-checkpoint), but not both!"
            )
        if not args.checkpoint and not args.resume_checkpoint:
            raise Exception(
                "You must specify a starting checkpoint (with --checkpoint) or a resume checkpoint (--resume-checkpoint)"
            )

        if args.checkpoint:
            hifi_gan = HifiGan.load_from_checkpoint(args.checkpoint)
        else:
            hifi_gan = HifiGan()

        # Fine-tuning overrides
        hifi_gan.finetune = True
        hifi_gan.lr = config["finetune_overrides"]["lr"]
        config["trainer"]["max_steps"] = config["finetune_overrides"]["max_steps"]

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
            ckpt_path=args.resume_checkpoint,
        )

    elif args.mode == "test":
        print("Testing")
    elif args.mode == "torchscript":
        print("Exporting torchscript")
