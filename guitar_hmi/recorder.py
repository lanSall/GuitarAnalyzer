"""Prompted recording sessions that produce labeled training clips.

Layout of the output folder:
    labels.csv                one row per saved clip (the training labels)
    clips/<session>/*.wav     one clip per pluck/strum, starting just before the onset
    sessions/<session>.wav    the full continuous recording, for re-segmenting later
    sessions/<session>.json   session settings and summary
"""
import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .analysis import ANALYSIS_WINDOW_S, PRE_ONSET_S, OnsetDetector, StringState, detect_pitches
from .guitar import NUM_FRETS, OPEN_MIDI, OPEN_NAMES, VOICINGS, note_name, string_number, tab_string
from .ui import STYLE, FretboardWidget

FIELDS = [
    "file", "session", "recorded_at", "kind", "chord", "tab", "string", "fret", "midi",
    "quality", "mistake_string", "technique",
    "fret_E", "fret_A", "fret_D", "fret_G", "fret_B", "fret_e",
    "onset_s", "duration_s", "sr", "peak", "qc_pass", "heard",
]

NOTE_TECHNIQUES = ["pick, medium", "pick, soft", "pick, hard", "finger, medium",
                   "finger, near the bridge", "pick, over the soundhole"]
CHORD_TECHNIQUES = ["downstroke, pick", "upstroke, pick", "downstroke, thumb",
                    "slow downstroke", "downstroke, hard", "downstroke, soft"]

MODES = {
    "notes": "Single notes",
    "chords": "Chords",
    "mistake_notes": "Mistakes: single notes (muted + buzz)",
    "mistake_chords": "Mistakes: chords (one bad string)",
}

PRE_ROLL_S = 0.1
ARM_DELAY_S = 0.35
COOLDOWN_S = 0.5
CLIP_PEAK = 0.99
QUIET_PEAK = 0.005
CHORD_MATCH = 0.6


@dataclass
class Prompt:
    kind: str  # "note" | "chord"
    frets: dict  # string index -> fret
    quality: str = "clean"  # clean | muted | buzz
    chord: str = ""
    mistake_string: int | None = None
    technique: str = ""
    take: int = 1
    takes: int = 1
    position: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.kind, tab_string(self.frets), self.quality)

    @property
    def title(self):
        if self.kind == "note":
            (s, f), = self.frets.items()
            where = "open" if f == 0 else f"fret {f}"
            return f"String {string_number(s)} ({OPEN_NAMES[s]})  ·  {where}  ·  {note_name(OPEN_MIDI[s] + f)}"
        return f"{self.chord}   {tab_string(self.frets)}"

    @property
    def instruction(self):
        bad = f"string {string_number(self.mistake_string)}" if self.mistake_string is not None else ""
        if self.kind == "note":
            return {
                "clean": f"Pluck once and let it ring  —  {self.technique}",
                "muted": "Pluck it, then DAMPEN it straight away (palm or a finger resting on it)",
                "buzz": "Fret it too lightly (or right on the fret wire) so it BUZZES",
            }[self.quality]
        return {
            "clean": f"Strum once and let it ring  —  {self.technique}",
            "muted": f"Strum it, but let {bad} be MUTED (lean a finger on it)",
            "buzz": f"Strum it, but make {bad} BUZZ (press it too lightly)",
        }[self.quality]


def read_labels(out_dir):
    path = os.path.join(out_dir, "labels.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_counts(rows):
    counts = {}
    for r in rows:
        if r.get("qc_pass") == "1":
            key = (r["kind"], r["tab"], r["quality"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_plan(mode, strings, fret_lo, fret_hi, chords, takes, counts, only_missing, shuffle, rng):
    groups = []
    if mode in ("notes", "mistake_notes"):
        qualities = ["clean"] if mode == "notes" else ["muted", "buzz"]
        for s in strings:
            for f in range(fret_lo, fret_hi + 1):
                for q in qualities:
                    groups.append(dict(kind="note", frets={s: f}, quality=q, mistake_string=None if q == "clean" else s))
    else:
        qualities = ["clean"] if mode == "chords" else ["muted", "buzz"]
        for name in chords:
            frets = {i: f for i, f in enumerate(VOICINGS[name]) if f is not None}
            for q in qualities:
                groups.append(dict(kind="chord", frets=frets, quality=q, chord=name))
    if shuffle:
        rng.shuffle(groups)

    plan = []
    for pos, g in enumerate(groups, 1):
        key = (g["kind"], tab_string(g["frets"]), g["quality"])
        n = max(takes - counts.get(key, 0), 0) if only_missing else takes
        techniques = NOTE_TECHNIQUES if g["kind"] == "note" else CHORD_TECHNIQUES
        played = sorted(g["frets"])
        for k in range(n):
            p = Prompt(take=k + 1, takes=n, position=pos, technique=techniques[k % len(techniques)], **g)
            if g["kind"] == "chord" and p.quality != "clean":
                p.mistake_string = played[k % len(played)]
            plan.append(p)
    return plan, len(groups)


class RecorderWindow(QMainWindow):
    def __init__(self, source, out_dir="dataset"):
        super().__init__()
        self.source = source
        self.sr = source.sr
        self.out_dir = os.path.abspath(out_dir)
        self.rng = np.random.default_rng()
        self.setWindowTitle("Guitar Analyzer — Training Data Recorder")
        self.setStyleSheet(STYLE + """
            QGroupBox { border: 1px solid #1f2937; border-radius: 6px; margin-top: 14px; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #94a3b8; }
            QPushButton { background: #1e293b; border: 1px solid #334155; border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background: #334155; }
            QPushButton:disabled { color: #475569; }
            QPushButton#primary { background: #2563eb; border-color: #3b82f6; font-weight: bold; }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background: #111827; border: 1px solid #334155; padding: 3px; }
        """)

        central = QWidget()
        row = QHBoxLayout(central)
        row.addWidget(self._build_setup(), 0)
        row.addWidget(self._build_session(), 1)
        self.setCentralWidget(central)

        self.onsets = OnsetDetector(self.sr)
        self.state = "setup"
        self.plan, self.index, self.saved = [], 0, []
        self.session = None
        self.session_file = None
        self.written = 0
        self.armed_at = 0
        self.capture_onset = None
        self.cooldown_until = 0.0
        self.peak_hold = 0.0
        self._target_states = []
        self.fretboard.quality_text = {"ringing": "play", "muted": "MUTE it", "buzz": "BUZZ it", "idle": "—"}

        for key, slot in (("Space", self._toggle_pause), ("S", self._skip_position), ("Backspace", self._undo)):
            QShortcut(QKeySequence(key), self, activated=slot)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._mode_changed()
        self._refresh_coverage()

    # ------------------------------------------------------------------ setup
    def _build_setup(self):
        box = QGroupBox("Session setup")
        box.setFixedWidth(340)
        form = QFormLayout(box)

        self.mode = QComboBox()
        for key, text in MODES.items():
            self.mode.addItem(text, key)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("What to record", self.mode)

        strings = QGridLayout()
        self.string_boxes = []
        for i in range(6):
            cb = QCheckBox(f"{string_number(i)}  {OPEN_NAMES[i]}")
            cb.setChecked(True)
            strings.addWidget(cb, i // 3, i % 3)
            self.string_boxes.append(cb)
        form.addRow("Strings", strings)

        frets = QHBoxLayout()
        self.fret_lo, self.fret_hi = QSpinBox(), QSpinBox()
        for sb, v in ((self.fret_lo, 0), (self.fret_hi, 5)):
            sb.setRange(0, NUM_FRETS)
            sb.setValue(v)
            frets.addWidget(sb)
        frets.insertWidget(1, QLabel("to"))
        form.addRow("Frets", frets)

        self.chord_list = QListWidget()
        self.chord_list.setFixedHeight(170)
        for name in VOICINGS:
            item = QListWidgetItem(f"{name:<7} {tab_string({i: f for i, f in enumerate(VOICINGS[name]) if f is not None})}")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in ("C", "G", "D", "A", "E", "Am", "Em", "Dm") else Qt.Unchecked)
            self.chord_list.addItem(item)
        form.addRow("Chords", self.chord_list)

        self.takes = QSpinBox()
        self.takes.setRange(1, 200)
        self.takes.setValue(15)
        form.addRow("Takes per position", self.takes)
        self.clip_len = QDoubleSpinBox()
        self.clip_len.setRange(0.5, 6.0)
        self.clip_len.setSingleStep(0.5)
        self.clip_len.setValue(2.0)
        self.clip_len.setSuffix(" s")
        form.addRow("Clip length", self.clip_len)

        self.only_missing = QCheckBox("Only record what's still missing")
        self.only_missing.setChecked(True)
        self.shuffle = QCheckBox("Shuffle position order")
        self.shuffle.setChecked(True)
        self.keep_failed = QCheckBox("Keep takes that fail the pitch check")
        for cb in (self.only_missing, self.shuffle, self.keep_failed):
            form.addRow(cb)

        out = QHBoxLayout()
        self.out_edit = QLineEdit(self.out_dir)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        out.addWidget(self.out_edit)
        out.addWidget(browse)
        form.addRow("Save to", out)

        self.coverage = QLabel("")
        self.coverage.setWordWrap(True)
        self.coverage.setStyleSheet("color: #94a3b8;")
        form.addRow(self.coverage)

        self.start_btn = QPushButton("Start session")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start_session)
        form.addRow(self.start_btn)
        self.setup_box = box
        return box

    def _build_session(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        self.title = QLabel("Set up a session on the left, then press Start")
        self.title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.instruction = QLabel("")
        self.instruction.setFont(QFont("Segoe UI", 14))
        self.instruction.setStyleSheet("color: #cbd5e1;")
        v.addWidget(self.title)
        v.addWidget(self.instruction)

        self.fretboard = FretboardWidget()
        self.fretboard.setMinimumHeight(260)
        v.addWidget(self.fretboard, 1)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress_text = QLabel("")
        self.progress_text.setStyleSheet("color: #94a3b8;")
        v.addWidget(self.progress)
        v.addWidget(self.progress_text)

        status_row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setFont(QFont("Segoe UI", 16, QFont.Bold))
        status_row.addWidget(self.status, 1)
        self.meter = QProgressBar()
        self.meter.setRange(0, 60)
        self.meter.setTextVisible(False)
        self.meter.setFixedWidth(220)
        self.meter_label = QLabel(self.source.label)
        self.meter_label.setStyleSheet("color: #94a3b8;")
        meter_col = QVBoxLayout()
        meter_col.addWidget(self.meter_label, alignment=Qt.AlignRight)
        meter_col.addWidget(self.meter, alignment=Qt.AlignRight)
        status_row.addLayout(meter_col)
        v.addLayout(status_row)

        buttons = QHBoxLayout()
        self.pause_btn = QPushButton("Pause  [Space]")
        self.skip_btn = QPushButton("Skip position  [S]")
        self.undo_btn = QPushButton("Undo last take  [Backspace]")
        self.end_btn = QPushButton("End session")
        for b, slot in ((self.pause_btn, self._toggle_pause), (self.skip_btn, self._skip_position),
                        (self.undo_btn, self._undo), (self.end_btn, self._end_session)):
            b.clicked.connect(slot)
            b.setEnabled(False)
            buttons.addWidget(b)
        v.addLayout(buttons)

        self.recent = QListWidget()
        self.recent.setFixedHeight(150)
        v.addWidget(self.recent)
        return panel

    def _mode_changed(self):
        notes = self.mode.currentData() in ("notes", "mistake_notes")
        for w in self.string_boxes + [self.fret_lo, self.fret_hi]:
            w.setEnabled(notes)
        self.chord_list.setEnabled(not notes)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Dataset folder", self.out_edit.text())
        if path:
            self.out_edit.setText(path)
            self._refresh_coverage()

    def _refresh_coverage(self):
        rows = read_labels(self.out_edit.text())
        good = [r for r in rows if r.get("qc_pass") == "1"]
        notes = sum(r["kind"] == "note" for r in good)
        self.coverage.setText(f"Dataset: {len(good)} clips ({notes} notes, {len(good) - notes} chords)"
                              if rows else "Dataset: empty")

    def start(self):
        self.source.start()
        self.meter_label.setText(self.source.label)
        self.timer.start(30)

    # ---------------------------------------------------------------- session
    def _start_session(self):
        self.out_dir = os.path.abspath(self.out_edit.text())
        mode = self.mode.currentData()
        strings = [i for i, cb in enumerate(self.string_boxes) if cb.isChecked()]
        chords = [self.chord_list.item(i).data(Qt.UserRole) for i in range(self.chord_list.count())
                  if self.chord_list.item(i).checkState() == Qt.Checked]
        counts = existing_counts(read_labels(self.out_dir))
        self.plan, n_positions = build_plan(
            mode, strings, self.fret_lo.value(), max(self.fret_hi.value(), self.fret_lo.value()), chords,
            self.takes.value(), counts, self.only_missing.isChecked(), self.shuffle.isChecked(), self.rng)
        if not self.plan:
            QMessageBox.information(self, "Nothing to record",
                                    "Every selected position already has enough takes (or nothing is selected).")
            return

        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.n_positions = n_positions
        for sub in ("clips", "sessions"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "clips", self.session), exist_ok=True)
        self.session_file = sf.SoundFile(os.path.join(self.out_dir, "sessions", f"{self.session}.wav"),
                                         "w", self.sr, 1, "PCM_24")
        self.written = self.source.ring.total
        self.session_meta = {
            "session": self.session, "started": datetime.now().isoformat(timespec="seconds"),
            "source": self.source.label, "sr": self.sr, "mode": mode, "strings": [string_number(s) for s in strings],
            "frets": [self.fret_lo.value(), self.fret_hi.value()], "chords": chords,
            "takes_per_position": self.takes.value(), "clip_length_s": self.clip_len.value(),
            "planned_takes": len(self.plan),
        }
        self.index, self.saved = 0, []
        self.setup_box.setEnabled(False)
        for b in (self.pause_btn, self.skip_btn, self.undo_btn, self.end_btn):
            b.setEnabled(True)
        self.recent.clear()
        self._arm()

    def _end_session(self, finished=False):
        if self.session_file:
            self.session_file.close()
            self.session_file = None
            self.session_meta.update(ended=datetime.now().isoformat(timespec="seconds"),
                                     saved_takes=len(self.saved), completed=finished)
            with open(os.path.join(self.out_dir, "sessions", f"{self.session}.json"), "w", encoding="utf-8") as f:
                json.dump(self.session_meta, f, indent=2)
        self.state = "setup"
        self.setup_box.setEnabled(True)
        for b in (self.pause_btn, self.skip_btn, self.undo_btn, self.end_btn):
            b.setEnabled(False)
        self.pause_btn.setText("Pause  [Space]")
        self.title.setText("Session complete" if finished else "Session ended")
        self.instruction.setText(f"{len(self.saved)} clips saved to {self.out_dir}")
        self._set_status("", "#e2e8f0")
        self.fretboard.set_state([], 0)
        self._refresh_coverage()

    def _arm(self):
        if self.index >= len(self.plan):
            self._end_session(finished=True)
            return
        p = self.plan[self.index]
        self.state = "listening"
        self.armed_at = self.source.ring.total + int(ARM_DELAY_S * self.sr)
        self.capture_onset = None
        self.title.setText(p.title)
        self.instruction.setText(p.instruction)
        self.progress.setRange(0, len(self.plan))
        self.progress.setValue(self.index)
        self.progress_text.setText(f"Take {p.take} of {p.takes}   ·   position {p.position} of {self.n_positions}"
                                   f"   ·   {len(self.saved)} saved this session   ·   {len(self.plan) - self.index} to go")
        self._set_status("● Listening — play now", "#4ade80")
        self._show_target(p)
        if hasattr(self.source, "request"):
            self.source.request(p.frets, {p.mistake_string: p.quality} if p.quality != "clean" else {})

    def _show_target(self, p):
        states = []
        for i in range(6):
            s = StringState(i)
            if i in p.frets:
                s.active, s.fret, s.midi, s.level_db = True, p.frets[i], OPEN_MIDI[i] + p.frets[i], -50.0
                s.quality = p.quality if (p.mistake_string == i and p.quality != "clean") else "ringing"
            states.append(s)
        self._target_states = states

    def _toggle_pause(self):
        if self.state in ("listening", "capturing"):
            self.state = "paused"
            self.pause_btn.setText("Resume  [Space]")
            self._set_status("❚❚ Paused", "#fbbf24")
        elif self.state == "paused":
            self.pause_btn.setText("Pause  [Space]")
            self._arm()

    def _skip_position(self):
        if self.state not in ("listening", "paused", "cooldown"):
            return
        pos = self.plan[self.index].position
        while self.index < len(self.plan) and self.plan[self.index].position == pos:
            self.index += 1
        self.pause_btn.setText("Pause  [Space]")
        self._arm()

    def _undo(self):
        if not self.saved or self.state == "setup":
            return
        entry = self.saved.pop()
        if os.path.exists(entry["path"]):
            os.remove(entry["path"])
        rows = [r for r in read_labels(self.out_dir) if r["file"] != entry["row"]["file"]]
        with open(os.path.join(self.out_dir, "labels.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.index = entry["index"]
        self.recent.insertItem(0, QListWidgetItem(f"↶ removed {entry['row']['file']}"))
        self.pause_btn.setText("Pause  [Space]")
        self._arm()

    # ------------------------------------------------------------------- loop
    def _tick(self):
        ring = self.source.ring
        total = ring.total
        x = ring.latest(int(0.05 * self.sr))
        rms_db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)
        self.peak_hold = max(self.peak_hold * 0.97, float(np.max(np.abs(x))) if len(x) else 0.0)
        self.meter.setValue(int(np.clip(rms_db + 60, 0, 60)))
        self.meter.setStyleSheet("QProgressBar::chunk { background: %s; }"
                                 % ("#ef4444" if self.peak_hold >= CLIP_PEAK else "#38bdf8"))

        if self.session_file and self.state != "paused":
            start = max(self.written, total - ring.size + 1)
            if total > start:
                self.session_file.write(ring.get_range(start, total))
            self.written = total
        elif self.state == "paused":
            self.written = total

        onsets = self.onsets.scan(ring)
        if self.state == "setup":
            return
        if self.state != "paused":
            self.fretboard.set_state(self._target_states, time.perf_counter())

        if self.state == "listening":
            onsets = [o for o in onsets if o >= self.armed_at]
            if onsets:
                self.capture_onset = onsets[0]
                self.state = "capturing"
                self._set_status("◉ Recording…", "#38bdf8")
        elif self.state == "capturing":
            if total >= self.capture_onset + int(self.clip_len.value() * self.sr):
                self._finish_take()
        elif self.state == "cooldown" and time.perf_counter() >= self.cooldown_until:
            self._arm()

    def _finish_take(self):
        p = self.plan[self.index]
        onset = self.capture_onset
        pre = int(PRE_ROLL_S * self.sr)
        clip = self.source.ring.get_range(onset - pre, onset + int(self.clip_len.value() * self.sr))
        a = int(PRE_ONSET_S * self.sr)
        heard = detect_pitches(clip[pre - a:pre - a + int(ANALYSIS_WINDOW_S * self.sr)], self.sr)
        expected = [OPEN_MIDI[i] + f for i, f in p.frets.items()]
        if p.kind == "note":
            # The prompt is the label. A piezo often reports open high E an octave
            # high (fret 12); that take is still the note they were asked to play.
            target = expected[0]
            passed = any(h - target in (0, 12, -12) for h in heard[:4])
        else:
            want, got = {m % 12 for m in expected}, {m % 12 for m in heard}
            passed = len(want & got) / max(len(want | got), 1) >= CHORD_MATCH
        peak = float(np.max(np.abs(clip)))
        heard_txt = " ".join(note_name(m) for m in heard) or "nothing"

        if peak >= CLIP_PEAK:
            return self._reject(f"✗ Input clipped — turn the gain down and play again (heard {heard_txt})")
        if peak < QUIET_PEAK:
            return self._reject("✗ Too quiet — check the pickup/gain and play again")
        if not passed and not self.keep_failed.isChecked():
            return self._reject(f"✗ Didn't match the target (heard {heard_txt}) — play it again")

        self._save(p, clip, pre, peak, passed, heard)
        self.index += 1
        self.state = "cooldown"
        self.cooldown_until = time.perf_counter() + COOLDOWN_S
        self._set_status(f"✓ Saved  (heard {heard_txt})" + ("" if passed else "  — kept despite failing check"),
                         "#4ade80" if passed else "#fbbf24")

    def _reject(self, message):
        self.recent.insertItem(0, QListWidgetItem(message))
        self.state = "cooldown"
        self.cooldown_until = time.perf_counter() + COOLDOWN_S + 0.5
        self._set_status(message, "#f87171")

    def _save(self, p, clip, pre, peak, passed, heard):
        n = len(self.saved) + 1
        tag = (f"s{string_number(next(iter(p.frets)))}f{next(iter(p.frets.values()))}" if p.kind == "note"
               else p.chord.replace("/", "-"))
        if p.quality != "clean":
            tag += f"_{p.quality}"
        name = f"{self.session}_{n:04d}_{tag}.wav"
        rel = os.path.join("clips", self.session, name).replace("\\", "/")
        path = os.path.join(self.out_dir, rel)
        sf.write(path, clip, self.sr, subtype="PCM_24")

        row = {
            "file": rel, "session": self.session, "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "kind": p.kind, "chord": p.chord, "tab": tab_string(p.frets),
            "string": string_number(next(iter(p.frets))) if p.kind == "note" else "",
            "fret": next(iter(p.frets.values())) if p.kind == "note" else "",
            "midi": OPEN_MIDI[next(iter(p.frets))] + next(iter(p.frets.values())) if p.kind == "note" else "",
            "quality": p.quality,
            "mistake_string": string_number(p.mistake_string) if p.mistake_string is not None else "",
            "technique": p.technique,
            "onset_s": round(pre / self.sr, 4), "duration_s": round(len(clip) / self.sr, 3), "sr": self.sr,
            "peak": round(peak, 4), "qc_pass": int(passed), "heard": " ".join(str(m) for m in heard),
        }
        for i, col in enumerate(("fret_E", "fret_A", "fret_D", "fret_G", "fret_B", "fret_e")):
            row[col] = p.frets.get(i, -1)
        labels = os.path.join(self.out_dir, "labels.csv")
        new = not os.path.exists(labels)
        with open(labels, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        self.saved.append({"path": path, "row": row, "index": self.index})
        self.recent.insertItem(0, QListWidgetItem(f"✓ {name}   heard {' '.join(note_name(m) for m in heard)}"))

    def _set_status(self, text, color):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    def closeEvent(self, event):
        if self.session_file:
            self._end_session()
        self.timer.stop()
        self.source.stop()
        super().closeEvent(event)
