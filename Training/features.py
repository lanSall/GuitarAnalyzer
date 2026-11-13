"""Model inputs computed from aligned clips (pluck at ONSET_S)."""
import math

import torch

from data import ONSET_S, SAMPLE_RATE

N_FFT = 8192
HOP = 2048
FEATURE_S = 1.0
F_MIN, F_MAX = 60.0, 12000.0
BANDS_PER_OCTAVE = 36

N_HARMONICS = 16
HARMONIC_FFT = 65536
EARLY_S = (0.0, 0.25)
LATE_S = (0.5, 0.75)
MAX_B = 4e-4
N_HARMONIC_FEATURES = 1 + 2 * N_HARMONICS


def log_filterbank():
    """(bands, FFT bins) triangular filters spaced evenly in pitch, so a semitone spans the same
    number of bands at every height. Low bands narrower than an FFT bin use the nearest bin."""
    freqs = torch.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    n = int(BANDS_PER_OCTAVE * math.log2(F_MAX / F_MIN))
    edges = F_MIN * 2 ** (torch.arange(n + 2) / BANDS_PER_OCTAVE)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = torch.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)).clamp(min=0)
    rows, nearest = torch.arange(n), (freqs - mid).abs().argmin(dim=1)
    bank[rows, nearest] = bank[rows, nearest].clamp(min=1)
    return bank / bank.sum(dim=1, keepdim=True)


def spectrogram(audio, bank):
    """Log-frequency spectrogram of the FEATURE_S after the pluck, (batch, 1, bands, frames).
    Values are dB below the clip's loudest band, mapped from -80..0 dB to 0..1."""
    start = int(ONSET_S * SAMPLE_RATE)
    segment = audio[:, start:start + int(FEATURE_S * SAMPLE_RATE) + N_FFT]
    spec = torch.stft(segment, N_FFT, HOP, window=torch.hann_window(N_FFT), center=False,
                      return_complex=True).abs()
    db = 20 * torch.log10(bank @ spec + 1e-8)
    db = db - db.amax(dim=(1, 2), keepdim=True)
    return (db.clamp(min=-80) / 80 + 1).unsqueeze(1)


def midi_hz(midi):
    return 440.0 * 2.0 ** ((midi.float() - 69.0) / 12.0)


def _harmonic_peaks(audio, f0, window_s):
    """Frequency (Hz) and level (dB) of the strongest peak near each harmonic of f0 in one
    window after the pluck, each (batch, N_HARMONICS). The search runs from 2% below h*f0 to
    2% above where a string with inharmonicity MAX_B would put it, which stops short of h+1."""
    start = int((ONSET_S + window_s[0]) * SAMPLE_RATE)
    n = int((window_s[1] - window_s[0]) * SAMPLE_RATE)
    spec = torch.fft.rfft(audio[:, start:start + n] * torch.hann_window(n), HARMONIC_FFT).abs()
    bin_hz = SAMPLE_RATE / HARMONIC_FFT
    h = torch.arange(1, N_HARMONICS + 1, dtype=torch.float32)
    lo = (0.98 * h * f0[:, None] / bin_hz).long()
    hi = (1.02 * h * torch.sqrt(1 + MAX_B * h * h) * f0[:, None] / bin_hz).long()
    width = int((hi - lo).max()) + 1
    idx = lo[..., None] + torch.arange(width)
    valid = idx <= hi[..., None]
    idx = idx.clamp(max=spec.shape[1] - 1)
    values = torch.gather(spec[:, None, :].expand(-1, N_HARMONICS, -1), 2, idx)
    values = values.masked_fill(~valid, 0)
    peak, where = values.max(dim=2)
    freq = (lo + where).float() * bin_hz
    return freq, 20 * torch.log10(peak + 1e-9)


def harmonic_features(audio, midi):
    """Tone features that separate strings playing the same pitch, (batch, N_HARMONIC_FEATURES):

    - inharmonicity B from a least-squares fit of (f_h / h)^2 = f0^2 (1 + B h^2), using
      harmonics within 40 dB of the loudest; wound and plain strings differ most here
    - each harmonic's early level in dB below the loudest harmonic
    - each harmonic's decay in dB per second from the early to the late window
    """
    f0 = midi_hz(midi)
    freq, early = _harmonic_peaks(audio, f0, EARLY_S)
    _, late = _harmonic_peaks(audio, f0, LATE_S)

    h = torch.arange(1, N_HARMONICS + 1, dtype=torch.float32)
    x, y = h * h, (freq / h) ** 2 / f0[:, None] ** 2
    w = (early > early.amax(dim=1, keepdim=True) - 40).float()
    sw = w.sum(dim=1)
    mx, my = (w * x).sum(dim=1) / sw, (w * y).sum(dim=1) / sw
    var = (w * (x - mx[:, None]) ** 2).sum(dim=1)
    slope = (w * (x - mx[:, None]) * (y - my[:, None])).sum(dim=1) / var.clamp(min=1e-9)
    intercept = my - slope * mx
    b = torch.where(sw >= 3, slope / intercept.clamp(min=1e-3), torch.zeros_like(slope))
    b = b.clamp(-MAX_B, MAX_B) * 1e4

    levels = early - early.amax(dim=1, keepdim=True)
    decay = (early - late) / (LATE_S[0] - EARLY_S[0])
    return torch.cat([b[:, None], levels, decay], dim=1)
