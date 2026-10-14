"""Turns the incoming audio stream into per-string state.

The HMI only relies on `analyzer.name`, `analyzer.process(ring) -> Frame` and the
StringState / Event / Frame types below, so a trained model can replace
DspAnalyzer without touching the UI.

DspAnalyzer is a hand-tuned signal-processing baseline. It finds pitches
reliably, but with a single pickup it cannot tell which string played a pitch
that exists in several places; it picks the most playable position (known
chord shapes first, otherwise the lowest-fret fingering). That decision is
exactly what the trained timbre model is meant to take over.
"""
import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .guitar import (
    HIGHEST_MIDI,
    LOWEST_MIDI,
    OPEN_MIDI,
    VOICINGS,
    midi_to_hz,
    name_chord,
    note_name,
    positions_for,
    string_number,
    tab_string,
)

# Thresholds are tuned on the simulator; expect to re-tune them on the real piezo.
ONSET_RISE_DB = 9.0
ONSET_FLOOR_DB = -62.0
REFRACTORY_S = 0.07
ANALYSIS_WINDOW_S = 0.1
PRE_ONSET_S = 0.005
RESTRIKE_RISE_DB = 3.0
REL_SALIENCE = 0.2
MAX_NOTES = 6
MUTE_CHECK_AGE_S = 0.32
MUTE_DROP_DB = 15.0
RELEASE_DROP_DB = 45.0
SILENCE_DB = -72.0
BUZZ_WINDOW_S = (0.13, 0.25)
TRACK_HARMONICS = 6
# Long enough to separate low notes a few semitones apart (E2-F2 is ~5 Hz).
TRACK_WINDOW_S = 0.17
BUZZ_BAND_HZ = (2500.0, 8000.0)
BUZZ_BAND_DB = -50.0
BUZZ_FLATNESS = 0.006
HARMONIC_REL_SALIENCE = 0.5
CHORD_SHAPE_BONUS = 0.02
ABSENT_RATIO = 0.08
ABSENT_PENALTY = 0.1
EXPLAIN_HARMONICS = 10
ABSENT_HARMONICS = 4
PEAK_FLOOR = 0.01


@dataclass
class StringState:
    index: int
    active: bool = False
    midi: int | None = None
    fret: int | None = None
    level_db: float = -120.0
    peak_db: float = -120.0
    onset: float = -1.0
    quality: str = "idle"  # idle | ringing | muted | buzz
    strike_id: int = 0
    mute_checked: bool = True
    harmonics: list = field(default_factory=list)
    shadowed: bool = False  # every tracked harmonic is shared with another ringing string


@dataclass
class Event:
    time: float
    detected_at: float
    kind: str  # "note" | "chord" | "mistake"
    label: str
    frets: dict = field(default_factory=dict)
    strike_ids: dict = field(default_factory=dict)
    quality: str = ""


@dataclass
class Frame:
    time: float
    strings: list
    events: list
    chord: str
    freqs: np.ndarray
    spectrum_db: np.ndarray
    input_db: float


def _pow2(n):
    return 1 << int(np.ceil(np.log2(max(n, 2))))


_windows = {}


def _spectrum(x, nfft):
    """Amplitude spectrum (a full-scale sine reads ~1.0 at its peak)."""
    win = _windows.get(len(x))
    if win is None:
        win = _windows[len(x)] = np.hanning(len(x)).astype(np.float32)
    return np.abs(np.fft.rfft(x * win, nfft)) * (2.0 / win.sum())


class _HarmonicBank:
    """Peak magnitudes at the harmonics of every playable note, for one FFT size."""

    def __init__(self, sr, nfft, window_len, n_harm=12, fmax=5000.0, tol=0.012):
        self.nfft = nfft
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        self.midis = np.arange(LOWEST_MIDI, HIGHEST_MIDI + 1)
        f0 = midi_to_hz(self.midis)
        fh = f0[:, None] * np.arange(1, n_harm + 1)[None, :]
        df = sr / nfft
        half = np.maximum(fh * tol, 1.5 * df)
        nbins = len(self.freqs)
        self.lo = np.clip(np.floor((fh - half) / df), 1, nbins - 2).astype(int)
        self.hi = np.clip(np.ceil((fh + half) / df) + 1, self.lo + 1, nbins - 1).astype(int)
        self.valid = fh < fmax
        self.weights = np.where(self.valid, (f0[:, None] + 27.0) / (fh + 320.0), 0.0)
        self._idx = np.stack([self.lo.ravel(), self.hi.ravel()], 1).ravel()
        # A Hann main lobe spans +-2 resolution bins of the unpadded window.
        lobe = np.maximum(fh * tol, 2.0 * sr / window_len)
        self.att_lo = np.clip(np.floor((fh - lobe) / df), 0, nbins - 1).astype(int)
        self.att_hi = np.clip(np.ceil((fh + lobe) / df) + 1, 0, nbins).astype(int)

    def peaks(self, mag):
        p = np.maximum.reduceat(mag, self._idx)[::2].reshape(self.lo.shape)
        return np.where(self.valid, p, 0.0)

    def salience(self, mag):
        return (self.peaks(mag) * self.weights).sum(1)

    def attenuate(self, mag, row, factor):
        for lo, hi, ok in zip(self.att_lo[row], self.att_hi[row], self.valid[row]):
            if ok:
                mag[lo:hi] *= factor


def _level_db(peaks, midi, harmonics=range(TRACK_HARMONICS)):
    return 20.0 * np.log10(peaks[midi - LOWEST_MIDI, list(harmonics)].sum() + 1e-9)


def _assign_harmonics(strings):
    """Track each ringing string on the harmonics no other ringing string shares."""
    active = [s for s in strings if s.active]
    for s in active:
        others = [midi_to_hz(o.midi) * np.arange(1, 2 * TRACK_HARMONICS + 1) for o in active if o is not s]
        others = np.concatenate(others) if others else np.empty(0)
        f = midi_to_hz(s.midi) * np.arange(1, TRACK_HARMONICS + 1)
        unique = [h for h in range(TRACK_HARMONICS) if not np.any(np.abs(others - f[h]) < 0.02 * f[h])]
        s.shadowed = not unique
        s.harmonics = unique or list(range(TRACK_HARMONICS))


class DspAnalyzer:
    name = "DSP baseline"

    def __init__(self, sr):
        self.sr = sr
        self.strings = [StringState(i) for i in range(6)]
        self.chord = ""
        self.track_len = _pow2(TRACK_WINDOW_S * sr)
        self.track_bank = _HarmonicBank(sr, self.track_len * 2, self.track_len)
        self.seg_len = int(ANALYSIS_WINDOW_S * sr)
        self.pre_len = int(PRE_ONSET_S * sr)
        self.seg_bank = _HarmonicBank(sr, _pow2(self.seg_len) * 4, self.seg_len)
        self.frame_len = max(int(0.004 * sr), 16)
        self.display_mask = (self.track_bank.freqs >= 40.0) & (self.track_bank.freqs <= 5000.0)
        self._pos = None
        self._recent = deque(maxlen=12)
        self._last_onset = -10 ** 12
        self._pending = []
        self._strikes = 0

    # ------------------------------------------------------------------ stream
    def process(self, ring):
        total = ring.total
        now = total / self.sr
        events = []
        self._scan_onsets(ring, total)

        due = sorted(p for p in self._pending if p[0] <= total)
        self._pending = [p for p in self._pending if p[0] > total]
        for _, kind, onset in due:
            ev = self._analyze_onset(ring, onset) if kind == "analyze" else self._check_buzz(ring, onset)
            if ev:
                ev.detected_at = now
                events.append(ev)

        x = ring.latest(self.track_len)
        mag = _spectrum(x, self.track_bank.nfft)
        events += self._track(mag, now)
        input_db = 20.0 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)
        spec_db = 20.0 * np.log10(mag[self.display_mask] + 1e-9)
        return Frame(now, self.strings, events, self.chord, self.track_bank.freqs[self.display_mask], spec_db, input_db)

    def _scan_onsets(self, ring, total):
        if self._pos is None:
            self._pos = total
        self._pos = max(self._pos, total - self.sr)
        n_frames = (total - self._pos) // self.frame_len
        if n_frames <= 0:
            return
        x = ring.get_range(self._pos - 1, self._pos + n_frames * self.frame_len)
        d = np.diff(x).reshape(n_frames, self.frame_len)
        energy_db = 10.0 * np.log10(np.mean(d * d, axis=1) + 1e-12)
        for k, e in enumerate(energy_db):
            start = self._pos + k * self.frame_len
            base = np.median(self._recent) if self._recent else e
            busy = any(kind == "analyze" for _, kind, _ in self._pending)
            if (
                e > ONSET_FLOOR_DB
                and e - base > ONSET_RISE_DB
                and start - self._last_onset > REFRACTORY_S * self.sr
                and not busy
            ):
                self._last_onset = start
                self._pending.append((start - self.pre_len + self.seg_len, "analyze", start))
            self._recent.append(e)
        self._pos += n_frames * self.frame_len

    # ------------------------------------------------------------- per strike
    def _analyze_onset(self, ring, onset):
        seg = ring.get_range(onset - self.pre_len, onset - self.pre_len + self.seg_len)
        if 20.0 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9) < SILENCE_DB + 10:
            return None
        mag = _spectrum(seg, self.seg_bank.nfft)
        post = self.seg_bank.peaks(mag)
        pre = self.track_bank.peaks(_spectrum(ring.get_range(onset - self.track_len, onset), self.track_bank.nfft))

        detected = self._multi_pitch(mag)
        for s in self.strings:
            if s.active and s.midi in detected and _level_db(post, s.midi) - _level_db(pre, s.midi) < RESTRIKE_RISE_DB:
                detected.remove(s.midi)
        if not detected:
            return None

        frets = self._assign(detected, mag)
        onset_s = onset / self.sr
        for s in self.strings:
            s.mute_checked = True
        strike_ids = {}
        for i, fret in frets.items():
            s = self.strings[i]
            self._strikes += 1
            s.active, s.fret, s.midi = True, fret, OPEN_MIDI[i] + fret
            s.onset, s.quality, s.strike_id, s.mute_checked = onset_s, "ringing", self._strikes, False
            strike_ids[i] = s.strike_id
        _assign_harmonics(self.strings)
        for i in frets:
            s = self.strings[i]
            s.level_db = s.peak_db = _level_db(post, s.midi, s.harmonics)
            s.mute_checked = s.shadowed

        midis = [OPEN_MIDI[i] + f for i, f in frets.items()]
        if len(frets) > 1:
            self.chord = name_chord(midis)
            label = f"{self.chord}  {tab_string(frets)}"
            kind = "chord"
        else:
            i, f = next(iter(frets.items()))
            self.chord = note_name(midis[0])
            label = f"string {string_number(i)} fret {f}  ({self.chord})"
            kind = "note"
        self._pending.append((onset + int(BUZZ_WINDOW_S[1] * self.sr), "buzz", onset))
        return Event(onset_s, 0.0, kind, label, frets, strike_ids)

    def _multi_pitch(self, mag):
        bank = self.seg_bank
        res = mag.copy()
        found = []
        first = None
        for _ in range(MAX_NOTES):
            sal = bank.salience(res)
            for m in found:
                sal[max(m - LOWEST_MIDI - 1, 0):m - LOWEST_MIDI + 2] = 0.0
            # A pitch sitting on a harmonic of a found note (octave, 12th, ...) needs
            # much more evidence; chord shapes restore genuine octave doublings later.
            harmonic_of_found = np.array([
                any(abs(12 * np.log2(h) - (c - m)) < 0.5 for m in found for h in range(2, 7))
                for c in bank.midis
            ])
            sal = np.where(harmonic_of_found & (sal < HARMONIC_REL_SALIENCE * (first or 0)), 0.0, sal)
            row = int(np.argmax(sal))
            first = sal[row] if first is None else first
            if sal[row] <= 0 or sal[row] < REL_SALIENCE * first:
                break
            found.append(LOWEST_MIDI + row)
            bank.attenuate(res, row, 0.15)
        return found

    def _assign(self, detected, mag):
        """Map detected pitches to strings: a known chord shape if one explains the spectrum
        at least as well as the detected notes do, else the most playable fingering."""
        free = self._fingering(detected)
        if len({m % 12 for m in detected}) < 2:
            return free
        peaks = self._spectral_peaks(mag)
        best, best_score = free, self._explain(peaks, [OPEN_MIDI[i] + f for i, f in free.items()]) - CHORD_SHAPE_BONUS
        for frets in VOICINGS.values():
            shape = {i: f for i, f in enumerate(frets) if f is not None}
            score = self._explain(peaks, [OPEN_MIDI[i] + f for i, f in shape.items()])
            if score > best_score:
                best, best_score = shape, score
        return best

    def _explain(self, peaks, notes):
        """Share of spectral-peak magnitude lying on the notes' harmonics, minus a penalty for
        each note whose own (unshared) harmonics have no peak in the spectrum."""
        freqs, amps = peaks
        if len(freqs) == 0:
            return 0.0
        h = np.arange(1, EXPLAIN_HARMONICS + 1)
        tol = 0.01 + 0.0015 * h
        combs = {m: midi_to_hz(m) * h for m in notes}

        def hits(comb, tolerance):
            return np.abs(freqs[:, None] - comb[None, :]) < tolerance[None, :] * comb[None, :]

        covered = np.zeros(len(freqs), bool)
        for comb in combs.values():
            covered |= hits(comb, tol).any(1)
        explained = amps[covered].sum() / amps.sum()

        strong = amps >= ABSENT_RATIO * amps.max()
        absent = 0
        for m, comb in combs.items():
            others = [c for o, c in combs.items() if o != m]
            if others:
                others = np.concatenate(others)
                own = ~np.any(np.abs(comb[:, None] - others[None, :]) < 0.02 * comb[:, None], axis=1)
            else:
                own = np.ones(len(comb), bool)
            own &= (comb < 5000.0) & (h <= ABSENT_HARMONICS)
            if own.any() and not hits(comb[own], np.full(own.sum(), 0.01))[strong].any():
                absent += 1
        return explained - ABSENT_PENALTY * absent

    def _spectral_peaks(self, mag):
        freqs = self.seg_bank.freqs
        is_peak = np.zeros(len(mag), bool)
        is_peak[1:-1] = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
        is_peak &= (freqs >= 60.0) & (freqs <= 5000.0) & (mag >= PEAK_FLOOR * mag.max())
        return freqs[is_peak], mag[is_peak]

    def _fingering(self, detected):
        notes = list(detected)
        while notes:
            best, best_cost = None, None
            for combo in itertools.product(*(positions_for(m) for m in notes)):
                if len({s for s, _ in combo}) < len(combo):
                    continue
                fretted = [f for _, f in combo if f > 0]
                span = max(fretted) - min(fretted) if fretted else 0
                cost = span + 10.0 * max(span - 4, 0) + 0.1 * sum(f for _, f in combo)
                if best_cost is None or cost < best_cost:
                    best, best_cost = dict(combo), cost
            if best:
                return best
            notes.pop()
        return {}

    def _check_buzz(self, ring, onset):
        struck = [s for s in self.strings if s.active and abs(s.onset - onset / self.sr) < 1e-9]
        if not struck:
            return None
        a, b = (int(onset + t * self.sr) for t in BUZZ_WINDOW_S)
        seg = ring.get_range(a, b)
        nfft = _pow2(len(seg))
        power = _spectrum(seg, nfft) ** 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.sr)
        band = (freqs >= BUZZ_BAND_HZ[0]) & (freqs <= BUZZ_BAND_HZ[1])
        full = (freqs >= 60.0) & (freqs <= BUZZ_BAND_HZ[1])
        band_db = 10.0 * np.log10(power[band].sum() / (power[full].sum() + 1e-20) + 1e-20)
        pb = power[band] + 1e-20
        flatness = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
        if band_db < BUZZ_BAND_DB or flatness < BUZZ_FLATNESS:
            return None
        if len(struck) == 1:
            s = struck[0]
            if s.quality == "ringing":
                s.quality = "buzz"
            label = f"BUZZ on string {string_number(s.index)} (fret {s.fret})"
        else:
            label = f"BUZZ in {self.chord} (string not identified)"
        return Event(onset / self.sr, 0.0, "mistake", label, {s.index: s.fret for s in struck},
                     {s.index: s.strike_id for s in struck}, "buzz")

    # ---------------------------------------------------------- continuous
    def _track(self, mag, now):
        events = []
        peaks = self.track_bank.peaks(mag)
        for s in self.strings:
            if not s.active:
                continue
            s.level_db = _level_db(peaks, s.midi, s.harmonics)
            age = now - s.onset
            if age < 0.12:
                s.peak_db = max(s.peak_db, s.level_db)
            silent = s.level_db < SILENCE_DB or s.level_db < s.peak_db - RELEASE_DROP_DB
            if not s.mute_checked and (age >= MUTE_CHECK_AGE_S or silent):
                s.mute_checked = True
                if silent or s.peak_db - s.level_db > MUTE_DROP_DB:
                    s.quality = "muted"
                    events.append(Event(s.onset, now, "mistake",
                                        f"MUTED string {string_number(s.index)} (fret {s.fret})",
                                        {s.index: s.fret}, {s.index: s.strike_id}, "muted"))
            if silent:
                s.active, s.quality = False, "idle"
                _assign_harmonics(self.strings)
        return events
