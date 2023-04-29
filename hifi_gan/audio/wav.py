from torch import Tensor
from torch.nn import functional as F

__HIFI_GAN_PAD = (1024 - 256) // 2


def pad_wav_generator(wav: Tensor):
    return F.pad(wav, (__HIFI_GAN_PAD, __HIFI_GAN_PAD))
