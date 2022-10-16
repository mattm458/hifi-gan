from os import path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(
        self,
        wav_dir: str,
        files: str,
        sample_rate: int = 22050,
        silence: float = 0.1,
        trim: bool = True,
        trim_frame_length=512,
    ):
        self.wav_dir = wav_dir
        self.files = files
        self.sample_rate = sample_rate

        self.silence_len = int(silence * sample_rate)
        self.trim = trim
        self.trim_frame_length = trim_frame_length

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Tensor:
        filename = self.files[i]

        wav, sr = librosa.load(path.join(self.wav_dir, filename))

        if sr != self.sample_rate:
            raise Exception(
                f"Sample rate of loaded WAV ({sr}) is not the same as configured sample rate ({self.sr}!"
            )

        # Trimming excess silence from the front and end of the wav file
        if self.trim:
            wav, _ = librosa.effects.trim(wav, frame_length=self.trim_frame_length)

        wav = np.pad(wav, (0, self.silence_len))

        mel_spectrogram = librosa.feature.melspectrogram(
            y=wav,
            sr=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            fmin=0.0,
            fmax=8000.0,
            n_mels=80,
            power=1,
            norm="slaney",
            pad_mode="reflect",
        )

        # Clamp to a small minimum value to avoid log(0)
        mel_spectrogram = np.clip(mel_spectrogram, a_min=1e-5, a_max=None)

        # Make it a log-Mel spectrogram
        mel_spectrogram = np.log(mel_spectrogram)

        # Swap axes so the spectrogram is (timesteps, n_mels)
        # instead of (n_mels, timesteps)
        mel_spectrogram = mel_spectrogram.swapaxes(0, 1)

        return torch.from_numpy(mel_spectrogram)
