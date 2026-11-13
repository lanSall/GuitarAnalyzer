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
ONSET_FLOOR_DB = -68.0
REFRACTORY_S = 0.07
ANALYSIS_WINDOW_S = 0.1
PRE_ONSET_S = 0.005
# A note must get louder at the pluck. The live 82 Hz drone otherwise looks like
# a permanent open low E, and its harmonics hide B and high E.
RESTRIKE_RISE_DB = 6.0
REL_SALIENCE = 0.35
MAX_NOTES = 6
MUTE_CHECK_AGE_S = 0.32
# dB drop from peak that counts as a dead string. High E on a soundhole piezo
# decays this much while still ringing; only a real mute falls farther.
MUTE_DROP_DB = [18.0, 20.0, 22.0, 26.0, 32.0, 40.0]
RELEASE_DROP_DB = 24.0
SILENCE_DB = -72.0
BUZZ_WINDOW_S = (0.22, 0.38)
TRACK_HARMONICS = 6
# Long enough to separate low notes a few semitones apart (E2-F2 is ~5 Hz).
TRACK_WINDOW_S = 0.17
BUZZ_BAND_HZ = (2500.0, 8000.0)
BUZZ_BAND_DB = -38.0
BUZZ_FLATNESS = 0.02
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


def _is_harmonic_of(higher, lower, cents=60.0):
    if higher <= lower:
        return False
    ratio = midi_to_hz(higher) / midi_to_hz(lower)
    nearest = round(ratio)
    if nearest < 2 or nearest > 6:
        return False
    return abs(1200.0 * np.log2(ratio / nearest)) < cents


def _without_harmonic_duplicates(found):
    """If two notes are harmonic (octave, 12th, …), keep the louder one.

    Always keeping the lower note hid open B and high E whenever the 82 Hz
    low-E drone was present. Chord shapes can still restore real doublings.
    """
    notes = [(int(m), float(s)) for m, s in found]
    drop = set()
    for i, (a, sa) in enumerate(notes):
        for b, sb in notes:
            if a == b:
                continue
            if _is_harmonic_of(a, b):
                drop.add(a if sa <= sb * 1.15 else b)
            elif _is_harmonic_of(b, a):
                drop.add(b if sb <= sa * 1.15 else a)
    kept = []
    for midi, _ in notes:
        if midi not in drop and midi not in kept:
            kept.append(midi)
    return kept


def _fold_down_octaves(bank, mag, found):
    """If 2f was picked as the note, step down when f0 or 3f of the lower octave exist.

    Open high E (330 Hz) on a piezo is often reported as fret 12 (660 Hz).
    The 3rd partial of E4 (~990 Hz) is not a harmonic of E5, so it is evidence
    the lower octave is the real string.
    """
    if not found:
        return found
    peaks = bank.peaks(mag)
    out = []
    for midi in found:
        cur = int(midi)
        while cur - 12 >= LOWEST_MIDI:
            lo = cur - 12
            hi_f0 = float(peaks[cur - LOWEST_MIDI, 0]) + 1e-12
            lo_f0 = float(peaks[lo - LOWEST_MIDI, 0])
            lo_h3 = float(peaks[lo - LOWEST_MIDI, 2])
            if lo_f0 >= 0.08 * hi_f0 or lo_h3 >= 0.10 * hi_f0:
                cur = lo
            else:
                break
        if cur not in out:
            out.append(cur)
    return out


def _multi_pitch(bank, mag):
    """Iteratively pick the strongest harmonic series and remove it, most salient first."""
    res = mag.copy()
    found = []
    first = None
    for _ in range(MAX_NOTES):
        sal = bank.salience(res)
        for m, _s in found:
            sal[max(m - LOWEST_MIDI - 1, 0):m - LOWEST_MIDI + 2] = 0.0
        picked = [m for m, _s in found]
        harmonic_of_found = np.array([
            any(_is_harmonic_of(c, m) for m in picked)
            for c in bank.midis
        ])
        sal = np.where(harmonic_of_found & (sal < HARMONIC_REL_SALIENCE * (first or 0)), 0.0, sal)
        row = int(np.argmax(sal))
        first = sal[row] if first is None else first
        if sal[row] <= 0 or sal[row] < REL_SALIENCE * first:
            break
        found.append((LOWEST_MIDI + row, float(sal[row])))
        bank.attenuate(res, row, 0.15)
    return _fold_down_octaves(bank, mag, _without_harmonic_duplicates(found))


_pitch_banks = {}


def detect_pitches(segment, sr):
    """Pitches (MIDI, most salient first) in a short segment that starts just before an onset."""
    key = (sr, len(segment))
    bank = _pitch_banks.get(key)
    if bank is None:
        bank = _pitch_banks[key] = _HarmonicBank(sr, _pow2(len(segment)) * 4, len(segment))
    return _multi_pitch(bank, _spectrum(np.asarray(segment, np.float32), bank.nfft))


class OnsetDetector:
    """Flags sudden rises in high-frequency energy (pluck attacks) in a RingBuffer stream."""

    def __init__(self, sr, rise_db=ONSET_RISE_DB, floor_db=ONSET_FLOOR_DB, refractory_s=REFRACTORY_S):
        self.sr = sr
        self.rise_db = rise_db
        self.floor_db = floor_db
        self.refractory = int(refractory_s * sr)
        self.frame_len = max(int(0.004 * sr), 16)
        self._pos = None
        self._recent = deque(maxlen=12)
        self._last_onset = -10 ** 12

    def scan(self, ring):
        """Onset sample indices found in audio that arrived since the last call."""
        total = ring.total
        if self._pos is None:
            self._pos = total
        self._pos = max(self._pos, total - self.sr)
        n_frames = (total - self._pos) // self.frame_len
        if n_frames <= 0:
            return []
        x = ring.get_range(self._pos - 1, self._pos + n_frames * self.frame_len)
        d = np.diff(x).reshape(n_frames, self.frame_len)
        energy_db = 10.0 * np.log10(np.mean(d * d, axis=1) + 1e-12)
        onsets = []
        for k, e in enumerate(energy_db):
            start = self._pos + k * self.frame_len
            base = np.median(self._recent) if self._recent else e
            if e > self.floor_db and e - base > self.rise_db and start - self._last_onset > self.refractory:
                self._last_onset = start
                onsets.append(start)
            self._recent.append(e)
        self._pos += n_frames * self.frame_len
        return onsets


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
        self.display_mask = (self.track_bank.freqs >= 40.0) & (self.track_bank.freqs <= 5000.0)
        # Strums arriving before the first strike's analysis window closes are merged into it.
        self.onsets = OnsetDetector(sr, refractory_s=max(REFRACTORY_S, ANALYSIS_WINDOW_S - PRE_ONSET_S))
        self._pending = []
        self._strikes = 0

    # ------------------------------------------------------------------ stream
    def process(self, ring):
        total = ring.total
        now = total / self.sr
        events = []
        for onset in self.onsets.scan(ring):
            self._pending.append((onset - self.pre_len + self.seg_len, "analyze", onset))

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

    # ------------------------------------------------------------- per strike
    def _analyze_onset(self, ring, onset):
        seg = ring.get_range(onset - self.pre_len, onset - self.pre_len + self.seg_len)
        if 20.0 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9) < SILENCE_DB + 10:
            return None
        mag = _spectrum(seg, self.seg_bank.nfft)
        post = self.seg_bank.peaks(mag)
        pre = self.track_bank.peaks(_spectrum(ring.get_range(onset - self.track_len, onset), self.track_bank.nfft))

        detected = [
            m for m in _multi_pitch(self.seg_bank, mag)
            if _level_db(post, m) - _level_db(pre, m) >= RESTRIKE_RISE_DB
        ]
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
        heard = " ".join(note_name(m) for m in detected)
        if len(frets) > 1:
            self.chord = name_chord(midis)
            label = f"{self.chord}  {tab_string(frets)}   heard {heard}"
            kind = "chord"
        else:
            i, f = next(iter(frets.items()))
            self.chord = note_name(midis[0])
            label = f"string {string_number(i)} fret {f}  ({self.chord})   heard {heard}"
            kind = "note"
        self._pending.append((onset + int(BUZZ_WINDOW_S[1] * self.sr), "buzz", onset))
        return Event(onset_s, 0.0, kind, label, frets, strike_ids)

    def _assign(self, detected, mag):
        """Map detected pitches to strings. Single notes stay on the lowest-fret
        fingering; library shapes are only tried for actual strums, and only if
        they contain every pitch class we heard."""
        free = self._fingering(detected)
        detected_pc = {m % 12 for m in detected}
        if len(detected_pc) < 2:
            return free
        peaks = self._spectral_peaks(mag)
        best, best_score = free, self._explain(peaks, [OPEN_MIDI[i] + f for i, f in free.items()])
        for frets in VOICINGS.values():
            shape = {i: f for i, f in enumerate(frets) if f is not None}
            notes = [OPEN_MIDI[i] + f for i, f in shape.items()]
            if not detected_pc.issubset({n % 12 for n in notes}):
                continue
            score = self._explain(peaks, notes) + CHORD_SHAPE_BONUS
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
                # Going quiet is not enough: treble notes on this pickup fade fast.
                if s.peak_db - s.level_db > MUTE_DROP_DB[s.index]:
                    s.quality = "muted"
                    events.append(Event(s.onset, now, "mistake",
                                        f"MUTED string {string_number(s.index)} (fret {s.fret})",
                                        {s.index: s.fret}, {s.index: s.strike_id}, "muted"))
            if silent:
                s.active, s.quality = False, "idle"
                _assign_harmonics(self.strings)
        return events
