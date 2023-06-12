from typing import Tuple

import lightning as pl
import torch
from hifi_gan.model.generator import Generator
from hifi_gan.model.period_discriminator import MultiPeriodDiscriminator
from hifi_gan.model.scale_discriminator import MultiScaleDiscriminator
from torch import Tensor
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        mean = torch.mean((1 - dg) ** 2)
        gen_losses.append(mean)
        loss += mean

    return loss, gen_losses


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


class HifiGan(pl.LightningModule):
    def __init__(self, lr=0.0002, lr_decay=0.999):
        super().__init__()

        self.lr = lr
        self.lr_decay = lr_decay
        self.finetune = False

        self.generator = Generator()
        self.multi_period_discriminator = MultiPeriodDiscriminator()
        self.multi_scale_discriminator = MultiScaleDiscriminator()

        self.automatic_optimization = False

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=None,
            n_mels=80,
            power=1,
            norm="slaney",
            mel_scale="slaney",
            pad_mode="reflect",
            center=False,
        )

        self.weight_norm()

    def weight_norm(self):
        self.generator.weight_norm()
        self.multi_period_discriminator.weight_norm()
        self.multi_scale_discriminator.weight_norm()

    def remove_weight_norm(self):
        self.generator.remove_weight_norm()
        self.multi_period_discriminator.remove_weight_norm()
        self.multi_scale_discriminator.remove_weight_norm()

    def forward(self, mel_spectrogram: Tensor) -> Tensor:
        return self.generator(mel_spectrogram)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=self.lr, betas=(0.8, 0.99)
        )
        discriminator_optimizer = torch.optim.AdamW(
            list(self.multi_period_discriminator.parameters())
            + list(self.multi_scale_discriminator.parameters()),
            lr=self.lr,
            betas=(0.8, 0.99),
        )

        if self.finetune:
            return generator_optimizer, discriminator_optimizer
        else:
            generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                generator_optimizer, gamma=self.lr_decay
            )

            discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                discriminator_optimizer, gamma=self.lr_decay
            )

            return {
                "optimizer": generator_optimizer,
                "lr_scheduler": {
                    "scheduler": generator_scheduler,
                    "interval": "step",
                    "frequency": 687,
                    "name": "generator_scheduler",
                },
            }, {
                "optimizer": discriminator_optimizer,
                "lr_scheduler": {
                    "scheduler": discriminator_scheduler,
                    "interval": "step",
                    "frequency": 687,
                    "name": "discriminator_scheduler",
                },
            }

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        mel_spectrogram, y, mel_spectrogram_y = batch

        generator_optimizer, discriminator_optimizer = self.optimizers()

        # Produce a waveform with the generator
        y_hat = self(mel_spectrogram)

        # Train the discriminators
        # =====================================================================
        y_df_hat_r, y_df_hat_g, _, _ = self.multi_period_discriminator(
            y, y_hat.detach()
        )
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        y_ds_hat_r, y_ds_hat_g, _, _ = self.multi_scale_discriminator(y, y_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_discriminator = loss_disc_s + loss_disc_f

        discriminator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss_discriminator)
        discriminator_optimizer.step()

        self.log(
            "train_discriminator_loss", loss_discriminator, on_epoch=True, on_step=True
        )

        # Train the generator
        # =====================================================================
        # Create a Mel spectrogram from the predicted waveform
        mel_spectrogram_y_hat = self.mel_spectrogram(y_hat).swapaxes(1, 2)

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(mel_spectrogram_y, mel_spectrogram_y_hat) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.multi_period_discriminator(
            y, y_hat
        )
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.multi_scale_discriminator(
            y, y_hat
        )
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)

        loss_generator = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        generator_optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss_generator)
        generator_optimizer.step()

        current_iteration = (
            self.trainer.num_training_batches * self.current_epoch
        ) + batch_idx

        if not self.finetune and current_iteration > 0 and current_iteration % 687 == 0:
            generator_scheduler, discriminator_scheduler = self.lr_schedulers()
            discriminator_scheduler.step()
            generator_scheduler.step()

        self.log("train_generator_loss", loss_generator, on_epoch=True, on_step=True)

        self.log(
            "loss",
            loss_generator + loss_discriminator,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        mel_spectrogram, y, mel_spectrogram_y = batch

        # Produce a waveform with the generator
        y_hat = self(mel_spectrogram)

        mel_spectrogram_y_hat = self.mel_spectrogram(y_hat).swapaxes(1, 2)

        loss = F.l1_loss(mel_spectrogram_y_hat, mel_spectrogram_y)

        self.log("val_loss", loss, on_epoch=True, on_step=False)

        return loss

    def prediction_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        mel_spectrogram, _, _ = batch

        return self.generator(mel_spectrogram)
