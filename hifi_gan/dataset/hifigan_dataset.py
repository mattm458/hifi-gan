import random
from os import path
from typing import List, Optional, Tuple

import torchaudio
from speech_utils.audio.transforms import HifiGanMelSpectrogram, TacotronMelSpectrogram
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
        finetune: bool = False,
        finetune_dir: Optional[str] = None
    ):
        self.wav_dir = wav_dir
        self.files = files

        self.sample_rate = sample_rate

        self.segment_size = segment_size
        self.hop_length = 256

        self.tacotron_mel_spectrogram = TacotronMelSpectrogram()
        self.hifi_gan_spectrogram = HifiGanMelSpectrogram()

        assert not finetune or (
            finetune and finetune_dir
        ), "If fine-tuning, a directory of fine-tune spectrograms is required!"

        self.finetune = finetune
        self.finetune_dir = finetune_dir

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        filename = self.files[i]

        wav, sr = torchaudio.load(path.join(self.wav_dir, filename))
        wav = wav[0]

        if sr != self.sample_rate:
            raise Exception(
                f"Sample rate of loaded WAV ({sr}) is not the same as configured sample rate ({self.sample_rate}!"
            )

        if self.finetune:
            if not self.finetune_dir:
                raise Exception("Missing fine_tune directory!")

            tacotron_mel_path = path.join(
                self.finetune_dir, f"{filename.replace('/', '_')}.np.npy"
            )
            tacotron_mel = torch.from_numpy(np.load(tacotron_mel_path))

            # Pad the wav using the same padding technique as the Torch stft function
            signal_dim = wav.dim()
            extended_shape = [1] * (3 - signal_dim) + list(wav.size())
            pad = int(self.tacotron_mel_spectrogram.n_fft // 2)
            wav_padded = F.pad(wav.view(extended_shape), [pad, pad], "reflect")
            wav_padded = wav_padded.view(wav_padded.shape[-signal_dim:])

            mel_start = random.randint(0, tacotron_mel.shape[0] - 29)
            mel_end = mel_start + 29

            wav_start = mel_start * self.tacotron_mel_spectrogram.hop_length
            wav_end = wav_start + self.segment_size

            wav = wav_padded[wav_start:wav_end]
            mel_spectrogram_X = tacotron_mel[mel_start:mel_end]
            mel_spectrogram_y = self.hifi_gan_spectrogram(wav_padded[wav_start:wav_end])

            return mel_spectrogram_X, wav, mel_spectrogram_y
        else:
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

            return mel_spectrogram_X, wav, mel_spectrogram_y
