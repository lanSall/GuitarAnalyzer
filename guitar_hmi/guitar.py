"""Guitar constants, note helpers and the chord library.

String indices run 0..5 from the low E (string 6) to the high E (string 1).
"""
import numpy as np

OPEN_MIDI = [40, 45, 50, 55, 59, 64]
OPEN_NAMES = ["E", "A", "D", "G", "B", "e"]
NUM_FRETS = 19
LOWEST_MIDI = OPEN_MIDI[0]
HIGHEST_MIDI = OPEN_MIDI[-1] + NUM_FRETS

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def note_name(midi, octave=True):
    name = NOTE_NAMES[int(midi) % 12]
    return f"{name}{int(midi) // 12 - 1}" if octave else name


def string_number(index):
    """Conventional string number (1 = high E) for a 0-based low-to-high index."""
    return 6 - index


def positions_for(midi):
    """All (string_index, fret) pairs that produce this MIDI note."""
    return [(s, midi - OPEN_MIDI[s]) for s in range(6) if 0 <= midi - OPEN_MIDI[s] <= NUM_FRETS]


def tab_string(frets_by_string):
    """Format {string_index: fret} as a low-to-high shape such as 'x32010'."""
    frets = [frets_by_string.get(i) for i in range(6)]
    parts = ["x" if f is None else str(f) for f in frets]
    return "-".join(parts) if any(f is not None and f >= 10 for f in frets) else "".join(parts)


# Common open/first-position voicings, low E to high E (None = not played).
VOICINGS = {
    "C": [None, 3, 2, 0, 1, 0],
    "C7": [None, 3, 2, 3, 1, 0],
    "D": [None, None, 0, 2, 3, 2],
    "D7": [None, None, 0, 2, 1, 2],
    "Dm": [None, None, 0, 2, 3, 1],
    "Dsus2": [None, None, 0, 2, 3, 0],
    "E": [0, 2, 2, 1, 0, 0],
    "E7": [0, 2, 0, 1, 0, 0],
    "Em": [0, 2, 2, 0, 0, 0],
    "Em7": [0, 2, 0, 0, 0, 0],
    "F": [1, 3, 3, 2, 1, 1],
    "Fmaj7": [None, None, 3, 2, 1, 0],
    "G": [3, 2, 0, 0, 0, 3],
    "G7": [3, 2, 0, 0, 0, 1],
    "A": [None, 0, 2, 2, 2, 0],
    "A7": [None, 0, 2, 0, 2, 0],
    "Am": [None, 0, 2, 2, 1, 0],
    "Am7": [None, 0, 2, 0, 1, 0],
    "Asus2": [None, 0, 2, 2, 0, 0],
    "B7": [None, 2, 1, 2, 0, 2],
    "Bm": [None, 2, 4, 4, 3, 2],
    "E5": [0, 2, 2, None, None, None],
    "A5": [None, 0, 2, 2, None, None],
}

CHORD_TEMPLATES = {
    "": {0, 4, 7},
    "m": {0, 3, 7},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "m7": {0, 3, 7, 10},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "5": {0, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "6": {0, 4, 7, 9},
    "m6": {0, 3, 7, 9},
    "add9": {0, 2, 4, 7},
}


def name_chord(midis):
    """Name the chord formed by a collection of MIDI notes (voicing independent)."""
    midis = list(midis)
    if not midis:
        return ""
    pcs = {m % 12 for m in midis}
    bass = min(midis) % 12
    if len(pcs) == 1:
        return note_name(midis[0], octave=False)
    matches = [
        (root, suffix)
        for root in range(12)
        for suffix, template in CHORD_TEMPLATES.items()
        if {(root + i) % 12 for i in template} == pcs
    ]
    if not matches:
        return " ".join(note_name(m, octave=False) for m in sorted(pcs))
    root, suffix = next(((r, s) for r, s in matches if r == bass), matches[0])
    name = NOTE_NAMES[root] + suffix
    return name if root == bass else f"{name}/{NOTE_NAMES[bass]}"
