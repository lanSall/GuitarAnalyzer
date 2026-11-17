"""Everything played this session as tab, kept in step with the analyzer's string choices, and
its export as ASCII tab or MusicXML (standard notation over tab; guitar_hmi/sheet.py prints it).

The gap between plucks is rounded to whole 8th notes on a 16th-note grid. The beat is the median
time between plucks, doubled or halved into BEAT_RANGE_S, so the tab reads as notes on beats; with
fewer than two notes it is DEFAULT_BEAT_S (120 BPM). Exports assume 4/4; the key is the major key
that fits the notes played.
"""
import statistics
from dataclasses import dataclass, field

from .guitar import NOTE_NAMES, OPEN_MIDI, OPEN_NAMES, string_number

DEFAULT_BEAT_S = 0.5
BEAT_RANGE_S = (0.4, 1.2)
STEPS_PER_BEAT = 4
QUANTUM_STEPS = 2  # plucks land on 8th notes; at playing speed, finer detail is mostly timing noise
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
        """(beat seconds, [(start step, length in steps, note)]). Each gap between plucks is
        rounded on its own, so small timing drift doesn't add up into odd rhythms. A note lasts
        until the next pluck unless it went quiet in under half that time."""
        beat = self.beat_seconds()
        step_s = beat / STEPS_PER_BEAT
        starts = [0] if self.notes else []
        for a, b in zip(self.notes, self.notes[1:]):
            gap = round((b.time - a.time) / (step_s * QUANTUM_STEPS)) * QUANTUM_STEPS
            starts.append(starts[-1] + max(QUANTUM_STEPS, gap))
        placed = []
        for k, (start, note) in enumerate(zip(starts, self.notes)):
            room = starts[k + 1] - start if k + 1 < len(starts) else LAST_NOTE_STEPS
            heard = round((note.end - note.time) / step_s) if note.end is not None else room
            placed.append((start, max(1, room if heard * 2 >= room else heard), note))
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

    def musicxml(self, last_bars=None):
        """The session as MusicXML; with last_bars, only that many bars from the end."""
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
        first = max(len(measures) - last_bars, 0) if last_bars else 0
        return _render_score(measures[first:], 60 / beat, self.key_fifths(), first + 1)

    def key_fifths(self):
        """Major key (as sharps > 0 / flats < 0) whose scale holds the most notes played,
        preferring fewer accidentals on a tie."""
        pcs = [(OPEN_MIDI[i] + f) % 12 for n in self.notes for i, f in n.frets.items()]

        def fit(fifths):
            tonic = (7 * fifths) % 12
            return sum((pc - tonic) % 12 in MAJOR_SCALE for pc in pcs), -abs(fifths)

        return max(range(-5, 7), key=fit) if pcs else 0


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
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
SHARP_SPELLING = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                  ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
FLAT_SPELLING = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
                 ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]
# Two staves: standard notation (staff 1, voice 1) and tab (staff 2, voice 5).
STAVES = ((1, 1), (2, 5))


def _pitch_xml(midi, fifths):
    step, alter = (FLAT_SPELLING if fifths < 0 else SHARP_SPELLING)[midi % 12]
    octave = midi // 12 - 1
    return (f"<pitch><step>{step}</step>{f'<alter>{alter}</alter>' if alter else ''}"
            f"<octave>{octave}</octave></pitch>")


def _note_xml(steps, note, ties, staff, voice, fifths):
    kind, dotted = TYPES[steps]
    dot = "<dot/>" if dotted else ""
    tail = f"<voice>{voice}</voice><type>{kind}</type>{dot}"
    if note is None and staff == 2:
        # Tab leaves silences blank.
        return f"<forward><duration>{steps}</duration><voice>{voice}</voice><staff>{staff}</staff></forward>"
    if note is None:
        return f"<note><rest/><duration>{steps}</duration>{tail}<staff>{staff}</staff></note>"
    out = []
    for k, (index, fret) in enumerate(sorted(note.frets.items())):
        tie = "".join(f'<tie type="{t}"/>' for t in ties)
        tied = "".join(f'<tied type="{t}"/>' for t in ties)
        technical = (f"<technical><string>{string_number(index)}</string><fret>{fret}</fret></technical>"
                     if staff == 2 else "")
        stem = "<stem>none</stem>" if staff == 2 else ""
        out.append(
            f"<note>{'<chord/>' if k else ''}{_pitch_xml(OPEN_MIDI[index] + fret, fifths)}"
            f"<duration>{steps}</duration>{tie}{tail}{stem}<staff>{staff}</staff>"
            f"<notations>{tied}{technical}</notations></note>")
    return "".join(out)


def _render_score(measures, bpm, fifths=0, first_number=1):
    tuning = "".join(
        f'<staff-tuning line="{i + 1}"><tuning-step>{NOTE_NAMES[m % 12][0]}</tuning-step>'
        f"<tuning-octave>{m // 12 - 1}</tuning-octave></staff-tuning>" for i, m in enumerate(OPEN_MIDI))
    attributes = (
        f"<attributes><divisions>{STEPS_PER_BEAT}</divisions><key><fifths>{fifths}</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>"
        # Guitar music is written an octave above how it sounds.
        '<clef number="1"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>'
        '<clef number="2"><sign>TAB</sign><line>5</line></clef>'
        f'<staff-details number="2"><staff-lines>6</staff-lines>{tuning}</staff-details></attributes>'
        f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
        f"<per-minute>{bpm:.0f}</per-minute></metronome></direction-type><staff>1</staff>"
        f'<sound tempo="{bpm:.0f}"/></direction>')
    body = []
    for k, measure in enumerate(measures):
        staves = ["".join(_note_xml(steps, note, ties, staff, voice, fifths) for steps, note, ties in measure)
                  for staff, voice in STAVES]
        back = f"<backup><duration>{sum(steps for steps, _, _ in measure)}</duration></backup>"
        body.append(f'<measure number="{first_number + k}">{attributes if k == 0 else ""}{back.join(staves)}</measure>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n'
            '<score-partwise version="4.0"><work><work-title>Guitar</work-title></work>'
            '<part-list><score-part id="P1"><part-name>Guitar</part-name></score-part></part-list>'
            f'<part id="P1">{"".join(body)}</part></score-partwise>\n')
