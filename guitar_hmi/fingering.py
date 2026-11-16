"""Choose a string for each note of a melody so the fretting hand stays playable.

Each note offers its possible placements ({string index: fret}) with a cost: -log of the string
model's probability, plus a little per fret so first position wins a tie. Moving from one
placement to the next costs more for a larger hand shift, a stretch beyond MAX_SPAN frets, and
crossing strings, and that cost fades over rests, when the hand has time to move. A Viterbi pass
over the last WINDOW notes picks the cheapest path, so a later note can revise an earlier guess.
Chords and notes the model does not cover take a single fixed placement and only give context.
"""
import bisect
import math
from dataclasses import dataclass

from .guitar import OPEN_MIDI

WINDOW = 8
MAX_SPAN = 4
MOVE_COST = 0.3     # per fret the hand shifts beyond one
STRETCH_COST = 2.0  # per fret beyond MAX_SPAN
STRING_COST = 0.1   # per string crossed
FRET_COST = 0.05    # per fret of the placement itself
REST_S = 1.5        # a gap this long cuts movement costs by a factor of e
MIN_PROB = 1e-3


@dataclass
class _Note:
    key: object
    onset: float
    options: list  # [(frets, cost)]
    choice: int | None = None

    @property
    def frets(self):
        return self.options[self.choice][0]


def _hand(frets):
    fretted = [f for f in frets.values() if f > 0]
    return (min(fretted), max(fretted)) if fretted else None


def move_cost(a, b, dt):
    """Cost of going from placement a to placement b, dt seconds later. Open strings need no hand."""
    cost = STRING_COST * abs(sum(a) / len(a) - sum(b) / len(b))
    ha, hb = _hand(a), _hand(b)
    if ha and hb:
        shift = abs((ha[0] + ha[1]) - (hb[0] + hb[1])) / 2
        span = max(ha[1], hb[1]) - min(ha[0], hb[0])
        cost += MOVE_COST * max(shift - 1, 0) + STRETCH_COST * max(span - MAX_SPAN, 0)
    return cost * math.exp(-max(dt, 0.0) / REST_S)


def note_options(midi, string_probs):
    """Placements for one pitch from {string index: probability}; strings missing from the dict
    are not candidates."""
    return [({i: midi - OPEN_MIDI[i]}, -math.log(max(p, MIN_PROB)) + FRET_COST * (midi - OPEN_MIDI[i]))
            for i, p in string_probs.items()]


class FingeringTracker:
    def __init__(self, window=WINDOW):
        self.window = window
        self._notes = []

    def add(self, key, onset, options):
        """Add a note with its [(frets, cost)] options, in any onset order. Returns
        [(key, frets)] for every note whose placement is new or changed."""
        note = _Note(key, onset, list(options))
        at = bisect.bisect_right([n.onset for n in self._notes], onset)
        self._notes.insert(at, note)
        changes = self._decode()
        del self._notes[:-(self.window + 1)]
        return changes

    def add_fixed(self, key, onset, frets):
        return self.add(key, onset, [(dict(frets), 0.0)])

    def _decode(self):
        notes = self._notes[-self.window:]
        anchor = self._notes[-self.window - 1] if len(self._notes) > self.window else None

        first = notes[0]
        total = [cost + (move_cost(anchor.frets, frets, first.onset - anchor.onset) if anchor else 0.0)
                 for frets, cost in first.options]
        back = []
        for prev, note in zip(notes, notes[1:]):
            dt = note.onset - prev.onset
            step, links = [], []
            for frets, cost in note.options:
                best = min(range(len(prev.options)),
                           key=lambda j: total[j] + move_cost(prev.options[j][0], frets, dt))
                step.append(total[best] + move_cost(prev.options[best][0], frets, dt) + cost)
                links.append(best)
            total = step
            back.append(links)

        choices = [min(range(len(total)), key=total.__getitem__)]
        for links in reversed(back):
            choices.append(links[choices[-1]])
        choices.reverse()

        changes = []
        for note, choice in zip(notes, choices):
            if note.choice != choice:
                note.choice = choice
                changes.append((note.key, note.frets))
        return changes
