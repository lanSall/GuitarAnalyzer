"""DspAnalyzer with trained models and a fingering tracker choosing strings for tab.

The DSP still detects plucks and pitch, and shows each note right away. About
250 ms after a pluck:

- single notes: the string model gives a probability for each string that can play that pitch
  in first position. The fingering tracker weighs those against how playable the last few notes
  are together, and may move this note or a recent one to another string.
- chords: the chord model (models/chord_model.pt, Training/train_chords.py) if one is saved,
  otherwise the chord fitter (guitar_hmi/chord_fit.py), replaces the DSP's note-by-note guess
  with a whole playable shape.

The fretboard, timeline and tab follow every change. Notes the string model was not trained on
(above its highest fret) keep the DSP's placement and only give the tracker context, as do
chords when no templates have been built.
"""
import dataclasses
import sys
from pathlib import Path

from .analysis import DspAnalyzer, Event, StringState, _assign_harmonics, _level_db, _spectrum
from .fingering import FingeringTracker, note_options
from .guitar import OPEN_MIDI, VOICINGS, name_chord, note_name, string_number, tab_string

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "string_model.pt"
DEFAULT_TEMPLATES = ROOT / "models" / "chord_templates.npz"
DEFAULT_CHORD_MODEL = ROOT / "models" / "chord_model.pt"
VOICING_NAMES = {}
for _name, _shape in VOICINGS.items():
    VOICING_NAMES.setdefault(tuple((i, f) for i, f in enumerate(_shape) if f is not None), _name)


def chord_name(frets):
    return VOICING_NAMES.get(tuple(sorted(frets.items()))) or name_chord(OPEN_MIDI[i] + f for i, f in frets.items())


class _FitterChords:
    label = "template fit"

    def __init__(self, path):
        from .chord_fit import ChordFitter

        self.fitter = ChordFitter(path, group="piezo")

    def detect(self, clip, onset_index):
        """(frets, how sure) or None."""
        fit = self.fitter.fit(clip, onset_index)
        return None if fit is None else (fit.frets, f"shape fit {fit.explained:.0%}")


class _ModelChords:
    label = "model"

    def __init__(self, path):
        from chords import ChordPredictor

        self.predictor = ChordPredictor(path)

    def detect(self, clip, onset_index):
        frets, _, joint = self.predictor.predict(clip)
        return (frets, f"chord model {joint:.0%}") if frets else None


class ModelAnalyzer(DspAnalyzer):
    name = "DSP + trained string model + fingering"

    def __init__(self, sr, model_path=DEFAULT_MODEL, templates_path=DEFAULT_TEMPLATES,
                 chord_model_path=DEFAULT_CHORD_MODEL):
        sys.path.insert(0, str(ROOT / "Training"))
        from data import SAMPLE_RATE
        from predictor import POST_SAMPLES, PRE_SAMPLES, StringPredictor

        if sr != SAMPLE_RATE:
            raise ValueError(f"The string model needs {SAMPLE_RATE} Hz audio; this source is {sr} Hz")
        super().__init__(sr)
        self.predictor = StringPredictor(model_path)
        self.pre, self.post = PRE_SAMPLES, POST_SAMPLES
        if chord_model_path and Path(chord_model_path).exists():
            self.chords = _ModelChords(chord_model_path)
        elif templates_path and Path(templates_path).exists():
            self.chords = _FitterChords(templates_path)
        else:
            self.chords = None
        if self.chords:
            self.name += f" + chord {self.chords.label}"
        self.fingering = FingeringTracker()
        self._waiting = []
        self._revisions = []

    def process(self, ring):
        frame = super().process(ring)
        due = [w for w in self._waiting if w[0] <= ring.total]
        self._waiting = [w for w in self._waiting if w[0] > ring.total]
        for _, onset, note_id, midi in due:
            if midi is None:
                self._apply_chord(ring, onset, note_id, frame.time)
            else:
                self._apply_model(ring, onset, note_id, midi)
        for ev in self._revisions:
            ev.detected_at = frame.time
        frame.events += self._revisions
        self._revisions = []
        return frame

    def _analyze_onset(self, ring, onset):
        ev = super()._analyze_onset(ring, onset)
        if ev is None:
            return ev
        note = self.tab.notes[-1]
        index, fret = next(iter(ev.frets.items()))
        midi = OPEN_MIDI[index] + fret
        if ev.kind == "note" and fret <= self.predictor.max_fret and midi in self.predictor.board.pitch_ids:
            self._waiting.append((onset + self.post, onset, note.id, midi))
        elif ev.kind == "chord" and self.chords:
            self._waiting.append((onset + self.post, onset, note.id, None))
        else:
            self._revise(self.fingering.add_fixed(note.id, note.time, note.frets), None)
        return ev

    # ------------------------------------------------------------ single notes
    def _apply_model(self, ring, onset, note_id, midi):
        probs = self.predictor.string_probs(ring.get_range(onset - self.pre, onset + self.post), midi)
        note = self.tab.get(note_id)
        if not probs:
            self._revise(self.fingering.add_fixed(note_id, note.time, note.frets), None)
            return
        by_index = {6 - number: p for number, p in probs.items()}
        changes = self.fingering.add(note_id, note.time, note_options(midi, by_index))
        if not any(key == note_id for key, _ in changes):
            changes.append((note_id, note.frets))
        self._revise(changes, (note_id, by_index, dict(note.frets)))

    def _revise(self, changes, scored):
        """Apply the tracker's placements to the tab and fretboard, with an event for the note
        the model just scored (scored = (note id, string probabilities, DSP placement)) and for
        each earlier note the tracker moved. Only single notes ever move."""
        for note_id, frets in changes:
            note = self.tab.get(note_id)
            moved = frets != note.frets
            if moved:
                self.tab.update(note_id, frets)
            index, fret = next(iter(frets.items()))
            where = f"string {string_number(index)} fret {fret}  ({note_name(OPEN_MIDI[index] + fret)})"
            if scored and note_id == scored[0]:
                _, by_index, dsp = scored
                (dsp_index, dsp_fret), = dsp.items()
                verdict = ("agrees with DSP" if index == dsp_index
                           else f"DSP said string {string_number(dsp_index)} fret {dsp_fret}")
                model_pick = max(by_index, key=by_index.get)
                if model_pick != index:
                    verdict += f"; fingering overrode the model's string {string_number(model_pick)}"
                label = f"{where}   model {by_index[index]:.0%}, {verdict}"
            elif moved:
                label = f"{where}   moved by fingering"
            else:
                continue
            sid = next(iter(note.strikes.values()))
            self._move_string(sid, frets)
            self._revisions.append(Event(note.time, 0.0, "note", label, dict(frets), {index: sid}))

    def _move_string(self, strike_id, frets):
        """Move a still-ringing single note to its new string. A string plays one note at a
        time, so whatever the target string held has stopped."""
        index, fret = next(iter(frets.items()))
        current = next((s for s in self.strings if s.active and s.strike_id == strike_id), None)
        if current is None or current.index == index:
            return
        self.strings[current.index] = StringState(current.index)
        self.strings[index] = dataclasses.replace(current, index=index, fret=fret)
        _assign_harmonics(self.strings)

    # ------------------------------------------------------------------ chords
    def _apply_chord(self, ring, onset, note_id, now):
        note = self.tab.get(note_id)
        found = self.chords.detect(ring.get_range(onset - self.pre, onset + self.post), self.pre)
        if found is None:
            self._revise(self.fingering.add_fixed(note_id, note.time, note.frets), None)
            return
        frets, sureness = found
        dsp, old_strikes = dict(note.frets), dict(note.strikes)
        if frets == dsp:
            strike_ids = old_strikes
        else:
            strike_ids = self._restrike(ring, note, frets, now)
            self.tab.update(note_id, frets, strike_ids)

        name = chord_name(frets)
        if note is self.tab.notes[-1]:
            self.chord = name
        verdict = "agrees with DSP" if frets == dsp else f"DSP said {tab_string(dsp)}"
        dropped = [sid for sid in old_strikes.values() if sid not in strike_ids.values()]
        self._revisions.append(Event(note.time, 0.0, "chord", f"{name}  {tab_string(frets)}   {sureness}, {verdict}",
                                     dict(frets), strike_ids, dropped=dropped))
        self._revise(self.fingering.add_fixed(note_id, note.time, frets), None)

    def _restrike(self, ring, note, frets, now):
        """Put the fretboard's strings in step with the chord's new shape. Strings already
        replaced by a later pluck are left alone. Returns {string index: strike id}."""
        old = dict(note.strikes)
        peaks = self.track_bank.peaks(_spectrum(ring.latest(self.track_len), self.track_bank.nfft))
        strike_ids = {}
        for i, sid in old.items():
            s = self.strings[i]
            if i not in frets and s.strike_id == sid:
                self.tab.release(sid, now)
                self.strings[i] = StringState(i)
        fresh = []
        for i, fret in frets.items():
            s = self.strings[i]
            if i in old and s.strike_id == old[i] and s.fret == fret:
                strike_ids[i] = old[i]
                continue
            self._strikes += 1
            strike_ids[i] = self._strikes
            if s.active and s.onset > note.time:
                continue
            self.strings[i] = StringState(i, True, OPEN_MIDI[i] + fret, fret, onset=note.time,
                                          quality="ringing", strike_id=self._strikes)
            fresh.append(i)
        _assign_harmonics(self.strings)
        for i in fresh:
            s = self.strings[i]
            s.level_db = s.peak_db = _level_db(peaks, s.midi, s.harmonics)
        return strike_ids
