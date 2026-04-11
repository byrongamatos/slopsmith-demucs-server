"""Wrapper to run demucs with a patched save_audio that uses soundfile instead of torchaudio.save.

torchaudio >= 2.11 requires torchcodec for saving, which has shared library issues.
This wrapper patches demucs.audio.save_audio to use soundfile directly.
"""

import sys
import soundfile as sf
import torch


def patched_save_audio(wav, path, samplerate=44100, bitrate=320, clip="rescale",
                       bits_per_sample=16, as_float=False, **kwargs):
    """Save audio tensor to WAV using soundfile."""
    path = str(path)
    if not path.endswith('.wav'):
        path = path + '.wav'
    # wav shape: (channels, samples) — soundfile expects (samples, channels)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if clip == "rescale":
        mx = wav.abs().max()
        if mx > 0:
            wav = wav / max(mx, 1e-8)
    elif clip == "clamp":
        wav = wav.clamp(-1, 1)
    data = wav.T.cpu().numpy()
    subtype = 'FLOAT' if as_float else 'PCM_16'
    sf.write(path, data, samplerate, subtype=subtype)


# Patch demucs before importing its main
import demucs.audio
demucs.audio.save_audio = patched_save_audio

# Now run demucs main
from demucs.separate import main
main()
