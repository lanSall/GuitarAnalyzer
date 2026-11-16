"""Which strings and frets a strum is playing, found by fitting it as a sum of string tones.

Picking pitches one at a time fails on chords: in a first-position shape almost every upper
note sits on a harmonic of a bass note. Instead every string/fret position up to MAX_FRET has a
template, the average spectrum of single-note recordings of it (Training/templates.py), and
whole playable shapes are scored by how well a non-negative mix of their templates reproduces
the strum's spectrum. Candidates are every library voicing plus the combinations of each
string's best-fitting frets.

Templates measured on the same guitar matter most: in tests, strums mixed from held-out piezo
takes were fitted about 70% exactly with piezo templates, against about 35% with templates
from other guitars.
"""
import itertools
from dataclasses import dataclass

import numpy as np

from .guitar import VOICINGS

WINDOW_S = (0.03, 0.25)  # after the onset; strings in a strum arrive within about 30 ms
NFFT = 32768
F_LO, F_HI = 70.0, 5000.0
BANDS_PER_OCTAVE = 48    # a multiple of 12, so one fret is a whole number of bands
COMPRESS = 0.5           # band magnitudes are raised to this power so quiet strings still count
MAX_FRET = 5
MAX_SPAN = 4
BEAM = 2                 # best frets per string carried into the shape search
ITERATIONS = 200
STRING_COST = 0.015      # per string in the shape, so extra strings must earn their place
VOICING_BONUS = 0.03
SILENT_SHARE = 0.03      # a shape string explaining less than this of the strum is reported silent
POSITIONS = [(s, f) for s in range(6) for f in range(MAX_FRET + 1)]


@dataclass
class ChordFit:
    frets: dict       # string index (0 = low E) -> fret
    shares: dict      # string index -> share of the strum's spectrum it explains
    explained: float  # share of the spectrum the whole shape explains
    silent: list      # shape strings that barely sound (muted or missed)


def playable(frets):
    """At least two strings, in one unbroken run (as in every library voicing; a strum hits
    neighboring strings), with fretted notes at most MAX_SPAN frets apart."""
    fretted = [f for f in frets.values() if f > 0]
    return (len(frets) >= 2 and max(frets) - min(frets) == len(frets) - 1
            and (not fretted or max(fretted) - min(fretted) <= MAX_SPAN))


class BandSpectrum:
    """Compressed log-frequency magnitude spectrum of the WINDOW_S after an onset."""

    def __init__(self, sr):
        self.start, self.stop = (int(t * sr) for t in WINDOW_S)
        self.window = np.hanning(self.stop - self.start).astype(np.float32)
        freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
        n = int(BANDS_PER_OCTAVE * np.log2(F_HI / F_LO))
        edges = F_LO * 2.0 ** (np.arange(n + 2) / BANDS_PER_OCTAVE)
        lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        self.bank = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None
                            ).astype(np.float32)

    def __call__(self, clip, onset_index):
        x = np.asarray(clip[onset_index + self.start:onset_index + self.stop], np.float32)
        x = np.pad(x, (0, len(self.window) - len(x)))
        return (self.bank @ np.abs(np.fft.rfft(x * self.window, NFFT))) ** COMPRESS


def shift_frets(spectrum, frets):
    """A template moved up (or down) by a number of frets."""
    bands = frets * BANDS_PER_OCTAVE // 12
    out = np.roll(spectrum, bands)
    if bands > 0:
        out[:bands] = 0
    elif bands < 0:
        out[bands:] = 0
    return out


class ChordFitter:
    def __init__(self, templates_path, group="piezo", sr=48000):
        data = np.load(templates_path)
        groups = list(data["groups"])
        w = data["spectra"][groups.index(group if group in groups else "pooled")].T
        self.spectrum = BandSpectrum(sr)
        if w.shape[0] != self.spectrum.bank.shape[0]:
            raise ValueError(f"{templates_path} uses other spectrum settings; rebuild it with Training\\templates.py")
        self.set_templates(w)
        self.column = {p: k for k, p in enumerate(POSITIONS)}
        self.voicings = {}
        for name, shape in VOICINGS.items():
            frets = {i: f for i, f in enumerate(shape) if f is not None}
            if all(f <= MAX_FRET for f in frets.values()):
                self.voicings.setdefault(tuple(sorted(frets.items())), name)

    def set_templates(self, w):
        """w: (bands, len(POSITIONS)) template spectra."""
        self.w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
        self.gram = self.w.T @ self.w

    def _nnls(self, masks, wx):
        """Non-negative activations (shapes, positions) restricted to each mask, by multiplicative
        updates on the normal equations."""
        a = masks * (wx.sum() / max(self.gram.sum(), 1e-9))
        for _ in range(ITERATIONS):
            a *= wx / (a @ self.gram + 1e-12)
        return a

    def fit(self, clip, onset_index):
        """Best playable shape for a strum whose onset is at sample onset_index of clip."""
        x = self.spectrum(clip, onset_index)
        xx = float(x @ x)
        if xx <= 0:
            return None
        wx = self.w.T @ x

        full = self._nnls(np.ones((1, len(POSITIONS))), wx)[0]
        choices = []
        for s in range(6):
            best = sorted(range(MAX_FRET + 1), key=lambda f: -full[self.column[(s, f)]])[:BEAM]
            choices.append([None] + best)
        shapes = {tuple(sorted(v.items())) for v in self._combos(choices) if playable(v)}
        shapes = sorted(shapes | set(self.voicings))

        masks = np.zeros((len(shapes), len(POSITIONS)))
        for k, shape in enumerate(shapes):
            masks[k, [self.column[p] for p in shape]] = 1
        a = self._nnls(masks, wx)
        residual = 1 - (2 * a @ wx - np.einsum("kp,pq,kq->k", a, self.gram, a)) / xx
        score = (residual + STRING_COST * masks.sum(axis=1)
                 - VOICING_BONUS * np.array([s in self.voicings for s in shapes]))
        k = int(np.argmin(score))
        shape = dict(shapes[k])
        explained = a[k] * wx
        shares = {s: float(explained[self.column[(s, f)]] / xx) for s, f in shape.items()}
        silent = [s for s, v in shares.items() if v < SILENT_SHARE]
        return ChordFit(shape, shares, float(1 - residual[k]), silent)

    def name(self, frets):
        return self.voicings.get(tuple(sorted(frets.items())))

    @staticmethod
    def _combos(choices):
        for pick in itertools.product(*choices):
            yield {s: f for s, f in enumerate(pick) if f is not None}
