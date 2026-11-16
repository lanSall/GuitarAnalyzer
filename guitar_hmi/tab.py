"""Everything played this session as tab, kept in step with the analyzer's string choices, and
its export as ASCII tab or MusicXML.

Timing is quantized to a 16th-note grid. The beat is the median time between plucks, doubled or
halved into BEAT_RANGE_S, so the tab reads as notes on beats; with fewer than two notes it is
DEFAULT_BEAT_S (120 BPM). Exports assume 4/4.
"""
import statistics
from dataclasses import dataclass, field

from .guitar import NOTE_NAMES, OPEN_MIDI, OPEN_NAMES, string_number

DEFAULT_BEAT_S = 0.5
BEAT_RANGE_S = (0.4, 1.2)
STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16
BARS_PER_LINE = 4
LAST_NOTE_STEPS = 4


@dataclass
class TabNote:
    id: int
    time: float
    frets: dict  # string index (0 = low E) -> fret
    strikes: dict = field(default_factory=dict)  # string index -> the analyzer's strike id
    end: float | None = None
    version: int = 0
    ringing: set = field(default_factory=set)


class TabSession:
    def __init__(self):
        self.notes = []
        self._by_id = {}
        self._by_strike = {}

    def add(self, time, frets, strike_ids):
        """Record a pluck; any earlier note still sounding on those strings ends here."""
        for prev in reversed(self.notes[-12:]):
            if prev.end is None and prev.frets.keys() & frets.keys():
                prev.end = time
        note = TabNote(len(self.notes) + 1, time, dict(frets), dict(strike_ids),
                       ringing=set(strike_ids.values()))
        self.notes.append(note)
        self._by_id[note.id] = note
        for sid in strike_ids.values():
            self._by_strike[sid] = note
        return note.id

    def get(self, note_id):
        return self._by_id[note_id]

    def update(self, note_id, frets, strike_ids=None):
        """A new placement for a note; strike_ids too when a chord now sounds on other strings."""
        note = self._by_id[note_id]
        note.frets = dict(frets)
        note.version += 1
        if strike_ids is not None:
            for sid in note.strikes.values():
                self._by_strike.pop(sid, None)
            note.strikes, note.ringing = dict(strike_ids), set(strike_ids.values())
            for sid in strike_ids.values():
                self._by_strike[sid] = note

    def release(self, strike_id, time):
        """A string went quiet; the note ends when all of its strings have."""
        note = self._by_strike.get(strike_id)
        if note is None or strike_id not in note.ringing:
            return
        note.ringing.discard(strike_id)
        if not note.ringing and note.end is None:
            note.end = time

    # ------------------------------------------------------------------ export
    def beat_seconds(self):
        gaps = [b.time - a.time for a, b in zip(self.notes, self.notes[1:]) if b.time - a.time > 0.05]
        if not gaps:
            return DEFAULT_BEAT_S
        beat = statistics.median(gaps)
        while beat < BEAT_RANGE_S[0]:
            beat *= 2
        while beat > BEAT_RANGE_S[1]:
            beat /= 2
        return beat

    def _grid(self):
        """(beat seconds, [(start step, length in steps, note)]), each note ending at the next
        pluck or when it went quiet, whichever is first."""
        beat = self.beat_seconds()
        step_s = beat / STEPS_PER_BEAT
        t0 = self.notes[0].time if self.notes else 0.0
        starts, last = [], -1
        for note in self.notes:
            last = max(round((note.time - t0) / step_s), last + 1)
            starts.append(last)
        placed = []
        for k, (start, note) in enumerate(zip(starts, self.notes)):
            room = starts[k + 1] - start if k + 1 < len(starts) else LAST_NOTE_STEPS
            heard = round((note.end - note.time) / step_s) if note.end is not None else room
            placed.append((start, max(1, min(room, heard)), note))
        return beat, placed

    def ascii(self):
        if not self.notes:
            return "(no notes)\n"
        beat, placed = self._grid()
        n_steps = placed[-1][0] + placed[-1][1]
        n_steps += -n_steps % STEPS_PER_BAR
        columns = [{} for _ in range(n_steps)]
        for start, _, note in placed:
            columns[start] = note.frets

        lines = [f"Tempo about {60 / beat:.0f} BPM (estimated), one dash per 16th note, 4/4", ""]
        bar_steps = STEPS_PER_BAR * BARS_PER_LINE
        for first in range(0, n_steps, bar_steps):
            rows = {i: f"{OPEN_NAMES[i]}|" for i in range(6)}
            for step in range(first, min(first + bar_steps, n_steps)):
                frets = columns[step]
                width = max((len(str(f)) for f in frets.values()), default=1)
                for i in range(6):
                    rows[i] += str(frets[i]).rjust(width, "-") if i in frets else "-" * width
                if (step + 1) % STEPS_PER_BAR == 0:
                    for i in range(6):
                        rows[i] += "|"
            lines += [rows[i] for i in reversed(range(6))] + [""]
        return "\n".join(lines)

    def musicxml(self):
        beat, placed = self._grid()
        events, pos = [], 0
        for start, length, note in placed:
            if start > pos:
                events.append((start - pos, None))
            events.append((length, note))
            pos = start + length
        pos_in_bar = pos % STEPS_PER_BAR
        if pos_in_bar or not events:
            events.append((STEPS_PER_BAR - pos_in_bar, None))

        # A note longer than one written value, or crossing a bar line, becomes tied pieces.
        measures, current, filled = [], [], 0
        for length, note in events:
            chain = []
            while length:
                take = min(length, STEPS_PER_BAR - filled)
                for steps in _split(take):
                    current.append((steps, note, []))
                    chain.append(current[-1])
                filled += take
                length -= take
                if filled == STEPS_PER_BAR:
                    measures.append(current)
                    current, filled = [], 0
            if note is not None:
                for j, (_, _, ties) in enumerate(chain):
                    ties += (["stop"] if j else []) + (["start"] if j < len(chain) - 1 else [])
        if current:
            measures.append(current)
        return _render_score(measures, 60 / beat)


def _split(length):
    """Durations in steps that add up to length, each one a written note value."""
    out = []
    while length:
        steps = next(d for d in TYPES if d <= length)
        out.append(steps)
        length -= steps
    return out


TYPES = {16: ("whole", False), 12: ("half", True), 8: ("half", False), 6: ("quarter", True),
         4: ("quarter", False), 3: ("eighth", True), 2: ("eighth", False), 1: ("16th", False)}


def _pitch_xml(midi):
    name = NOTE_NAMES[midi % 12]
    alter = "<alter>1</alter>" if name.endswith("#") else ""
    return f"<pitch><step>{name[0]}</step>{alter}<octave>{midi // 12 - 1}</octave></pitch>"


def _note_xml(steps, note, ties):
    kind, dotted = TYPES[steps]
    dot = "<dot/>" if dotted else ""
    if note is None:
        return f"<note><rest/><duration>{steps}</duration><voice>1</voice><type>{kind}</type>{dot}</note>"
    out = []
    for k, (index, fret) in enumerate(sorted(note.frets.items())):
        tie = "".join(f'<tie type="{t}"/>' for t in ties)
        tied = "".join(f'<tied type="{t}"/>' for t in ties)
        out.append(
            f"<note>{'<chord/>' if k else ''}{_pitch_xml(OPEN_MIDI[index] + fret)}"
            f"<duration>{steps}</duration>{tie}<voice>1</voice><type>{kind}</type>{dot}"
            f"<notations>{tied}<technical><string>{string_number(index)}</string>"
            f"<fret>{fret}</fret></technical></notations></note>")
    return "".join(out)


def _render_score(measures, bpm):
    tuning = "".join(
        f'<staff-tuning line="{i + 1}"><tuning-step>{NOTE_NAMES[m % 12][0]}</tuning-step>'
        f"<tuning-octave>{m // 12 - 1}</tuning-octave></staff-tuning>" for i, m in enumerate(OPEN_MIDI))
    attributes = (
        f"<attributes><divisions>{STEPS_PER_BEAT}</divisions><key><fifths>0</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>TAB</sign><line>5</line></clef>"
        f"<staff-details><staff-lines>6</staff-lines>{tuning}</staff-details></attributes>"
        f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
        f"<per-minute>{bpm:.0f}</per-minute></metronome></direction-type><sound tempo=\"{bpm:.0f}\"/></direction>")
    body = []
    for number, measure in enumerate(measures, 1):
        notes = "".join(_note_xml(steps, note, ties) for steps, note, ties in measure)
        body.append(f'<measure number="{number}">{attributes if number == 1 else ""}{notes}</measure>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n'
            '<score-partwise version="4.0"><work><work-title>Guitar tab</work-title></work>'
            '<part-list><score-part id="P1"><part-name>Guitar</part-name></score-part></part-list>'
            f'<part id="P1">{"".join(body)}</part></score-partwise>\n')
