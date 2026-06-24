"""Route XTTS audio IO through soundfile (torchaudio.load is broken here: torch 2.12 vs
torchaudio 2.11 vs torchcodec 0.14 ABI skew). Only the file-read is replaced; resample and all
tensor math stay on torchaudio. Import this module before constructing any Xtts model.
"""
import soundfile as sf
import torch, torchaudio
import TTS.tts.models.xtts as _xtts


def _load_audio(audiopath, sampling_rate):
    data, lsr = sf.read(str(audiopath), dtype="float32", always_2d=True)  # (N, C)
    audio = torch.from_numpy(data.T.copy())                               # (C, N)
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    audio.clip_(-1, 1)
    return audio


_xtts.load_audio = _load_audio
