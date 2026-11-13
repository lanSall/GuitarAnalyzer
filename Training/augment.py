"""Random variations of training clips, so the model learns the string and not the guitar.

Every function takes and returns a (batch, samples) tensor of aligned clips (pluck at ONSET_S),
except spec_augment, which works on spectrograms.
"""
import math

import torch

from data import ONSET_S, SAMPLE_RATE

MAX_CENTS = 15.0
MAX_SHIFT_S = 0.010
EQ_DB = 6.0
EQ_POINTS = 8
SNR_DB = (25.0, 50.0)
HUM_DB = (-45.0, -25.0)
P_WARP, P_EQ, P_NOISE, P_HUM = 0.8, 0.8, 0.7, 0.3


def _chosen(n, p):
    return torch.rand(n) < p


def warp(audio):
    """Detune by up to MAX_CENTS and move the pluck by up to MAX_SHIFT_S, with linear
    interpolation. Small enough that the note stays the labeled semitone."""
    n, length = audio.shape
    use = _chosen(n, P_WARP)
    ratio = torch.where(use, 2 ** ((torch.rand(n) * 2 - 1) * MAX_CENTS / 1200), torch.ones(n))
    shift = torch.where(use, (torch.rand(n) * 2 - 1) * MAX_SHIFT_S * SAMPLE_RATE, torch.zeros(n))
    onset = ONSET_S * SAMPLE_RATE
    t = torch.arange(length, dtype=torch.float32)
    src = onset + (t - onset - shift[:, None]) * ratio[:, None]
    i0 = src.floor().long()
    frac = src - i0
    inside = (i0 >= 0) & (i0 < length - 1)
    i0 = i0.clamp(0, length - 2)
    out = torch.gather(audio, 1, i0) * (1 - frac) + torch.gather(audio, 1, i0 + 1) * frac
    return torch.where(inside, out, torch.zeros_like(out))


def random_eq(audio):
    """A smooth random gain curve of up to +/-EQ_DB, linear in log frequency from 50 Hz to
    20 kHz, imitating a different guitar body or pickup."""
    n, length = audio.shape
    gains = (torch.rand(n, EQ_POINTS) * 2 - 1) * EQ_DB * _chosen(n, P_EQ)[:, None]
    freqs = torch.fft.rfftfreq(length, 1 / SAMPLE_RATE).clamp(min=50, max=20000)
    pos = torch.log2(freqs / 50) / math.log2(20000 / 50) * (EQ_POINTS - 1)
    i0 = pos.floor().long().clamp(max=EQ_POINTS - 2)
    frac = pos - i0
    curve_db = gains[:, i0] * (1 - frac) + gains[:, i0 + 1] * frac
    return torch.fft.irfft(torch.fft.rfft(audio) * 10 ** (curve_db / 20), n=length)


def add_noise(audio):
    """White or pink noise at SNR_DB below the note, and sometimes 60 Hz mains hum with harmonics."""
    n, length = audio.shape
    start = int(ONSET_S * SAMPLE_RATE)
    level = audio[:, start:start + SAMPLE_RATE].pow(2).mean(dim=1).sqrt()[:, None]

    noise = torch.randn(n, length)
    pink = torch.rand(n) < 0.5
    shaped = torch.fft.rfft(noise) / torch.fft.rfftfreq(length, 1 / SAMPLE_RATE).clamp(min=20).sqrt()
    noise = torch.where(pink[:, None], torch.fft.irfft(shaped, n=length), noise)
    noise = noise / noise.pow(2).mean(dim=1, keepdim=True).sqrt()
    snr = SNR_DB[0] + torch.rand(n, 1) * (SNR_DB[1] - SNR_DB[0])
    audio = audio + noise * level * 10 ** (-snr / 20) * _chosen(n, P_NOISE)[:, None]

    t = torch.arange(length) / SAMPLE_RATE
    hum = sum(torch.sin(2 * math.pi * 60 * k * t + torch.rand(n, 1) * 2 * math.pi) / k for k in range(1, 6))
    hum_db = HUM_DB[0] + torch.rand(n, 1) * (HUM_DB[1] - HUM_DB[0])
    return audio + hum * level * 10 ** (hum_db / 20) * _chosen(n, P_HUM)[:, None]


def augment(audio):
    out = add_noise(random_eq(warp(audio)))
    return out / out.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)


def spec_augment(spec, max_bands=12, max_frames=3):
    """Blank one random band range and one random time range per spectrogram, set to the floor."""
    spec = spec.clone()
    n, _, bands, frames = spec.shape
    for i in range(n):
        b = int(torch.randint(0, max_bands + 1, ()))
        b0 = int(torch.randint(0, bands - b + 1, ()))
        f = int(torch.randint(0, max_frames + 1, ()))
        f0 = int(torch.randint(0, frames - f + 1, ()))
        spec[i, :, b0:b0 + b] = 0
        spec[i, :, :, f0:f0 + f] = 0
    return spec
