import random
from os import path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from speech_utils.audio.transforms import HifiGanMelSpectrogram, TacotronMelSpectrogram
from speech_utils.preprocessing.feature_extraction import extract_features
from speech_utils.preprocessing.scaler import Scaler
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from hifi_gan.audio.wav import pad_wav_generator


class HifiGanDataset(Dataset):
    def __init__(
        self,
        wav_dir: str,
        files: List[str],
        sample_rate: int = 22050,
        segment_size: int = 8192,
        trim: bool = False,
        trim_frame_length=512,
        extract_features=None,
        scaler: Scaler = None,
    ):
        self.wav_dir = wav_dir
        self.files = files

        self.sample_rate = sample_rate

        self.segment_size = segment_size
        self.hop_length = 256

        self.trim = trim
        self.trim_frame_length = trim_frame_length

        self.tacotron_mel_spectrogram = TacotronMelSpectrogram()
        self.hifi_gan_spectrogram = HifiGanMelSpectrogram()

        self.extract_features = extract_features
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        filename = self.files[i]

        wav, sr = torchaudio.load(path.join(self.wav_dir, filename))
        wav = wav[0]

        if sr != self.sample_rate:
            raise Exception(
                f"Sample rate of loaded WAV ({sr}) is not the same as configured sample rate ({self.sr}!"
            )

        # Trimming excess silence from the front and end of the wav file
        if self.trim:
            wav, _ = librosa.effects.trim(
                wav.numpy(), frame_length=self.trim_frame_length
            )
            wav = torch.from_numpy(wav)

        # Choose a random segment from the wav data.
        if len(wav) >= self.segment_size:
            max_wav_start = len(wav) - self.segment_size
            wav_start = random.randint(0, max_wav_start)
            wav = wav[wav_start : wav_start + self.segment_size]

        else:
            # If the wav data is too short (unlikely in most TTS datasets),
            # pad it to get the correct segment size
            wav = F.pad(wav, (0, self.segment_size - len(wav)))

        wav_tacotron = pad_wav_generator(wav)
        mel_spectrogram_X = self.tacotron_mel_spectrogram(wav_tacotron)[2:-2]
        mel_spectrogram_y = self.hifi_gan_spectrogram(wav)

        if self.extract_features is not None:
            features_dict = extract_features(
                wav_data=wav.numpy(), sample_rate=self.sample_rate
            )
            features_scaled = self.scaler.transform(
                np.array([features_dict[f] for f in self.extract_features])
            ).values

            features_scaled = torch.from_numpy(features_scaled)
            features_clipped = torch.clamp(features_scaled, min=-1, max=1)

            return (
                mel_spectrogram_X,
                wav,
                mel_spectrogram_y,
                features_clipped,
            )

        return mel_spectrogram_X, wav, mel_spectrogram_y
