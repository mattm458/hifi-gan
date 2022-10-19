import random
from os import path
from typing import List, Tuple

import librosa
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


class HifiGanDataset(Dataset):
    def __init__(
        self,
        wav_dir: str,
        files: List[str],
        segment_size: int = 8192,
        sample_rate: int = 22050,
        silence: float = 0.1,
        trim: bool = True,
        trim_frame_length=512,
    ):
        self.wav_dir = wav_dir
        self.files = files

        self.segment_size = segment_size
        self.hop_length = 256

        self.sample_rate = sample_rate

        self.silence_len = int(silence * sample_rate)
        self.trim = trim
        self.trim_frame_length = trim_frame_length

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=self.hop_length,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            power=1,
            norm="slaney",
            mel_scale="slaney",
            center=False,
        )

        self.mel_spectrogram_y = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=self.hop_length,
            f_min=0.0,
            f_max=None,
            n_mels=80,
            power=1,
            norm="slaney",
            mel_scale="slaney",
            center=False,
        )

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

        # Pad with silence at the end
        # Note: For TTS applications with attention, this only appears to work if there
        # is an end token appended to the input text. If there is no end token, the TTS
        # may struggle to learn alignments because the silence is not associated with
        # any input value.
        wav = F.pad(wav, (0, self.silence_len))

        # Choose a random segment from the wav data.
        if len(wav) >= self.segment_size:
            max_wav_start = len(wav) - self.segment_size
            wav_start = random.randint(0, max_wav_start)
            wav = wav[wav_start : wav_start + self.segment_size]

        else:
            # If the wav data is too short (unlikely in most TTS datasets),
            # pad it to get the correct segment size
            wav = F.pad(wav, (0, self.segment_size - len(wav)))

        wav_mel = F.pad(
            wav.unsqueeze(0),
            (int((1024 - 256) / 2), int((1024 - 256) / 2)),
            mode="reflect",
        ).squeeze(0)

        mel_spectrogram_X = self.mel_spectrogram(wav_mel)

        mel_spectrogram_y = self.mel_spectrogram_y(wav_mel)

        # Clamp to a small minimum value to avoid log(0)
        mel_spectrogram_X = torch.clamp(mel_spectrogram_X, min=1e-5, max=None)

        # Make it a log-Mel spectrogram
        mel_spectrogram_X = torch.log(mel_spectrogram_X)

        # Swap axes so the spectrogram is (timesteps, n_mels)
        # instead of (n_mels, timesteps)
        mel_spectrogram_X = mel_spectrogram_X.swapaxes(0, 1)
        mel_spectrogram_y = mel_spectrogram_y.swapaxes(0, 1)

        return mel_spectrogram_X, wav, mel_spectrogram_y
