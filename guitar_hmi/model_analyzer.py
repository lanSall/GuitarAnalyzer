"""DspAnalyzer with the trained string model choosing the string for single notes.

The DSP still detects plucks, pitch, mutes and buzz, and shows each note right away. The
model needs 1.2 s of audio after the pluck, so it confirms or corrects the string about
1.2 s later, moving the note on the fretboard if it disagrees. Chords stay with the DSP.
"""
import dataclasses
import sys
from pathlib import Path

from .analysis import DspAnalyzer, Event, StringState, _assign_harmonics
from .guitar import OPEN_MIDI, note_name, string_number

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "string_model.pt"


class ModelAnalyzer(DspAnalyzer):
    name = "DSP + trained string model"

    def __init__(self, sr, model_path=DEFAULT_MODEL):
        sys.path.insert(0, str(ROOT / "Training"))
        from data import SAMPLE_RATE
        from predictor import POST_SAMPLES, PRE_SAMPLES, StringPredictor

        if sr != SAMPLE_RATE:
            raise ValueError(f"The string model needs {SAMPLE_RATE} Hz audio; this source is {sr} Hz")
        super().__init__(sr)
        self.predictor = StringPredictor(model_path)
        self.pre, self.post = PRE_SAMPLES, POST_SAMPLES
        self._waiting = []

    def process(self, ring):
        frame = super().process(ring)
        due = [w for w in self._waiting if w[0] <= ring.total]
        self._waiting = [w for w in self._waiting if w[0] > ring.total]
        for _, onset, strike_id, index, fret in due:
            ev = self._apply_model(ring, onset, strike_id, index, fret)
            if ev:
                ev.detected_at = frame.time
                frame.events.append(ev)
        return frame

    def _analyze_onset(self, ring, onset):
        ev = super()._analyze_onset(ring, onset)
        if ev and ev.kind == "note":
            index, fret = next(iter(ev.frets.items()))
            self._waiting.append((onset + self.post, onset, ev.strike_ids[index], index, fret))
        return ev

    def _apply_model(self, ring, onset, strike_id, dsp_index, dsp_fret):
        midi = OPEN_MIDI[dsp_index] + dsp_fret
        result = self.predictor.predict(ring.get_range(onset - self.pre, onset + self.post), midi)
        if result is None:
            return None
        number, probability = result
        index = 6 - number
        fret = midi - OPEN_MIDI[index]

        # A string plays one note at a time, so whatever the target string held has stopped.
        current = next((s for s in self.strings if s.active and s.strike_id == strike_id), None)
        if current is not None and current.index != index:
            self.strings[current.index] = StringState(current.index)
            self.strings[index] = dataclasses.replace(current, index=index, fret=fret)
            _assign_harmonics(self.strings)

        verdict = ("agrees with DSP" if index == dsp_index
                   else f"DSP said string {string_number(dsp_index)} fret {dsp_fret}")
        label = f"string {number} fret {fret}  ({note_name(midi)})   model {probability:.0%}, {verdict}"
        return Event(onset / self.sr, 0.0, "note", label, {index: fret}, {index: strike_id})
