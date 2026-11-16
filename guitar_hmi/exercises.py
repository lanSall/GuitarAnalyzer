"""Let-ring exercises: known note sequences for scoring notes plucked over strings that still ring.

Each note is (string number, fret) with string 1 = high E, the same convention as labels.csv.
"""
from .guitar import OPEN_MIDI

EXERCISES = {
    "open_strings": (
        "Open strings low to high and back",
        [(6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
    ),
    "c_arpeggio": (
        "C chord shape x32010, one string at a time, up and back",
        [(5, 3), (4, 2), (3, 0), (2, 1), (1, 0), (2, 1), (3, 0), (4, 2)],
    ),
    "g_d_arpeggios": (
        "G shape 320003 low to high, then D shape xx0232 low to high",
        [(6, 3), (5, 2), (4, 0), (3, 0), (2, 0), (1, 3), (4, 0), (3, 2), (2, 3), (1, 2)],
    ),
    "octaves": (
        "Octaves over a ringing root: E E E, A A A, D D",
        [(6, 0), (4, 2), (1, 0), (5, 0), (3, 2), (1, 5), (4, 0), (2, 3)],
    ),
}


def midi_of(string, fret):
    return OPEN_MIDI[6 - string] + fret


def ascii_tab(notes):
    """Six-line tab of a note sequence, high E on top."""
    lines = {s: f"{'eBGDAE'[s - 1]}|-" for s in range(1, 7)}
    for string, fret in notes:
        for s in lines:
            lines[s] += (str(fret) if s == string else "-" * len(str(fret))) + "--"
    return "\n".join(lines[s] + "|" for s in range(1, 7))
