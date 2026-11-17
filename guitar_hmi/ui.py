"""Real-time HMI: fretboard, per-string timeline, scrolling tab beside a dark Verovio score,
spectrum and event log."""
import math
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .analysis import MODES
from .guitar import NUM_FRETS, OPEN_NAMES, midi_to_hz, note_name, string_number

TAB_DIR = Path(__file__).resolve().parents[1] / "tabs"
TAB_COLOR, TAB_REVISED_COLOR = "#e2e8f0", "#38bdf8"

QUALITY_COLORS = {"ringing": "#4ade80", "muted": "#f87171", "buzz": "#fb923c", "idle": "#64748b"}
QUALITY_TEXT = {"ringing": "ringing", "muted": "MUTED", "buzz": "BUZZ", "idle": "idle"}
STRING_COLORS = ["#ef4444", "#f59e0b", "#eab308", "#22c55e", "#3b82f6", "#a855f7"]
GAUGES = [3.4, 2.8, 2.3, 1.8, 1.4, 1.1]
TIMELINE_S = 8.0
TICK_MS = 16
AXIS_W = 56
SHEET_BARS = 8
SHEET_REFRESH_S = 0.3

STYLE = """
QWidget { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI'; }
QListWidget { background: #111827; border: 1px solid #1f2937; font-family: Consolas; font-size: 12px; }
QProgressBar { background: #1f2937; border: none; height: 10px; }
QProgressBar::chunk { background: #38bdf8; }
"""


def _norm_level(db):
    return float(np.clip((db + 70.0) / 60.0, 0.0, 1.0))


class FretboardWidget(QWidget):
    LABEL_W = 70
    PANEL_W = 240

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.strings = []
        self.now = 0.0
        self.quality_text = QUALITY_TEXT

    def set_state(self, strings, now):
        self.strings = strings
        self.now = now
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        x0, x1 = self.LABEL_W + 24, w - self.PANEL_W - 12
        top, bottom = 34, h - 40
        rows = [top + (5 - i) * (bottom - top) / 5 for i in range(6)]
        scale = (x1 - x0) / (1 - 2 ** (-NUM_FRETS / 12))
        fx = [x0 + scale * (1 - 2 ** (-n / 12)) for n in range(NUM_FRETS + 1)]

        p.fillRect(QRectF(x0, top - 16, x1 - x0, bottom - top + 32), QColor("#3f2a1d"))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#d6c7a1"))
        mid, gap = (top + bottom) / 2, (bottom - top) / 5
        for n in (3, 5, 7, 9, 15, 17, 19):
            p.drawEllipse(QPointF((fx[n - 1] + fx[n]) / 2, mid), 5, 5)
        cx12 = (fx[11] + fx[12]) / 2
        p.drawEllipse(QPointF(cx12, mid - gap), 5, 5)
        p.drawEllipse(QPointF(cx12, mid + gap), 5, 5)

        p.setPen(QPen(QColor("#b8b8b8"), 2))
        for n in range(1, NUM_FRETS + 1):
            p.drawLine(QPointF(fx[n], top - 16), QPointF(fx[n], bottom + 16))
        p.setPen(QPen(QColor("#f1ead2"), 6))
        p.drawLine(QPointF(x0, top - 16), QPointF(x0, bottom + 16))

        small = QFont("Segoe UI", 8)
        p.setFont(small)
        p.setPen(QColor("#94a3b8"))
        for n in range(1, NUM_FRETS + 1):
            p.drawText(QRectF(fx[n - 1], bottom + 18, fx[n] - fx[n - 1], 16), Qt.AlignCenter, str(n))

        for st in self.strings:
            self._draw_string(p, st, rows[st.index], fx, x0, x1, w)

    def _draw_string(self, p, st, y, fx, x0, x1, w):
        i = st.index
        active = st.active and st.fret is not None
        color = QColor(QUALITY_COLORS[st.quality if active else "idle"])
        fret = st.fret if active else 0

        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.setPen(QColor("#e2e8f0"))
        p.drawText(QRectF(6, y - 12, self.LABEL_W, 24), Qt.AlignVCenter | Qt.AlignLeft,
                   f"{string_number(i)}   {OPEN_NAMES[i]}")

        start_x = fx[fret] if fret else x0
        if fret:
            p.setPen(QPen(QColor("#8b8f96"), GAUGES[i]))
            p.drawLine(QPointF(x0, y), QPointF(start_x, y))

        amp = 11.0 * _norm_level(st.level_db) if active else 0.0
        path = QPainterPath(QPointF(start_x, y))
        wobble = math.sin(2 * math.pi * (6.0 + 0.7 * i) * self.now)
        for k in range(1, 61):
            path.lineTo(start_x + (x1 - start_x) * k / 60, y + amp * wobble * math.sin(math.pi * k / 60))
        p.setBrush(Qt.NoBrush)
        if active and self.now - st.onset < 0.18:
            glow = QColor(color)
            glow.setAlpha(90)
            p.setPen(QPen(glow, GAUGES[i] + 8))
            p.drawPath(path)
        p.setPen(QPen(color if active else QColor("#8b8f96"), GAUGES[i]))
        p.drawPath(path)

        if active:
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            if fret == 0:
                p.setPen(QPen(color, 2.5))
                p.drawEllipse(QPointF(x0 - 13, y), 8, 8)
            else:
                cx = (fx[fret - 1] + fx[fret]) / 2
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawEllipse(QPointF(cx, y), 12, 12)
                p.setPen(QColor("#0f172a"))
                p.drawText(QRectF(cx - 12, y - 12, 24, 24), Qt.AlignCenter, str(fret))

        px = w - self.PANEL_W + 4
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.setPen(color if active else QColor("#64748b"))
        text = f"{note_name(st.midi):<4} fret {st.fret}" if active else "—"
        p.drawText(QRectF(px, y - 22, 130, 20), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRectF(px + 130, y - 22, 100, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   self.quality_text[st.quality if active else "idle"])
        bar_w = self.PANEL_W - 24
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1f2937"))
        p.drawRect(QRectF(px, y + 2, bar_w, 5))
        if active:
            p.setBrush(color)
            p.drawRect(QRectF(px, y + 2, bar_w * _norm_level(st.level_db), 5))


class SheetView(QWidget):
    """The PDF's Verovio engraving, in the tab panel's dark colours. Right-aligned so new bars
    stay in view, like the tab scrolling to now."""

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: #111827;")
        self.pixmap = None

    def sizeHint(self):
        return QSize(400, 180)

    def set_png(self, png):
        pix = QPixmap()
        if pix.loadFromData(png):
            self.pixmap = pix
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setClipRect(self.rect())
        p.fillRect(self.rect(), QColor("#111827"))
        if self.pixmap is None or self.pixmap.isNull():
            p.setPen(QColor("#64748b"))
            p.drawText(self.rect(), Qt.AlignCenter, "Sheet music appears here as you play\n(same engraving as Export PDF)")
            return
        h = max(self.height() - 8, 1)
        w = max(int(self.pixmap.width() * h / max(self.pixmap.height(), 1)), 1)
        scaled = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        room = max(self.width() - 8, 1)
        if scaled.width() > room:
            scaled = scaled.copy(scaled.width() - room, 0, room, scaled.height())
        p.drawPixmap(4, 4, scaled)


def _engrave_live(musicxml):
    from guitar_hmi.sheet import render_live_png

    return render_live_png(musicxml)


class MainWindow(QMainWindow):
    def __init__(self, source, analyzer):
        super().__init__()
        self.source = source
        self.analyzer = analyzer
        self.setWindowTitle("Guitar String Monitor")
        self.setStyleSheet(STYLE)
        pg.setConfigOptions(antialias=True, background="#111827", foreground="#cbd5e1")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addLayout(self._build_header())
        self.fretboard = FretboardWidget()
        layout.addWidget(self.fretboard, 3)
        layout.addWidget(self._build_timeline(), 3)
        split = QSplitter(Qt.Horizontal)
        tab = self._build_tab()
        tab.setMinimumWidth(320)
        split.addWidget(tab)
        self.sheet = SheetView()
        split.addWidget(self.sheet)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 1000])
        layout.addWidget(split, 3)
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_spectrum(), 1)
        self.log = QListWidget()
        bottom.addWidget(self.log, 1)
        layout.addLayout(bottom, 2)
        self.setCentralWidget(central)

        self.hist_t = deque(maxlen=int(TIMELINE_S * 1000 / TICK_MS) + 60)
        self.hist_lv = deque(maxlen=self.hist_t.maxlen)
        self.strike_items = {}
        self.tab_items = {}
        self.sheet_pool = None
        self.sheet_job = None
        self.sheet_shown = None
        self.sheet_sent_at = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ----------------------------------------------------------------- layout
    def _build_header(self):
        header = QHBoxLayout()
        self.chord_label = QLabel("—")
        self.chord_label.setFont(QFont("Segoe UI", 34, QFont.Bold))
        self.chord_label.setMinimumWidth(220)
        header.addWidget(self.chord_label)

        mid = QVBoxLayout()
        self.detail_label = QLabel("Play open strings one at a time. Low E should peak near the 82 Hz tick.")
        self.detail_label.setFont(QFont("Segoe UI", 13))
        mid.addWidget(self.detail_label)
        header.addLayout(mid, 1)

        right = QVBoxLayout()
        self.source_label = QLabel(self.source.label)
        self.source_label.setStyleSheet("color: #94a3b8;")
        right.addWidget(self.source_label, alignment=Qt.AlignRight)
        analyzer_label = QLabel(f"Analyzer: {self.analyzer.name}")
        analyzer_label.setStyleSheet("color: #94a3b8;")
        right.addWidget(analyzer_label, alignment=Qt.AlignRight)
        self.meter = QProgressBar()
        self.meter.setRange(0, 60)
        self.meter.setTextVisible(False)
        self.meter.setFixedWidth(220)
        right.addWidget(self.meter, alignment=Qt.AlignRight)
        export = QPushButton("Export tab")
        export.setToolTip("Save everything played so far as ASCII tab (.txt), MusicXML (.musicxml) and "
                          "sheet music with notation over tab (.pdf)")
        export.setStyleSheet("QPushButton { background: #1f2937; border: 1px solid #334155; padding: 4px 12px; }")
        export.clicked.connect(self._export_tab)
        right.addWidget(export, alignment=Qt.AlignRight)
        self.mode_box = QComboBox()
        self.mode_box.setToolTip("Chords or Picking skips the note-or-chord guess when you know what you are playing")
        self.mode_box.setStyleSheet("QComboBox { background: #1f2937; border: 1px solid #334155; padding: 3px 8px; }")
        for key, text in MODES.items():
            self.mode_box.addItem(text, key)
        self.mode_box.setCurrentIndex(list(MODES).index(self.analyzer.mode))
        self.mode_box.currentIndexChanged.connect(self._set_mode)
        right.addWidget(self.mode_box, alignment=Qt.AlignRight)
        header.addLayout(right)
        return header

    def _set_mode(self, _index):
        self.analyzer.mode = self.mode_box.currentData()
        self._log(f"Mode: {self.mode_box.currentText()}", "#94a3b8")

    def _build_timeline(self):
        tl = self.timeline = pg.PlotWidget()
        tl.setMouseEnabled(False, False)
        tl.hideButtons()
        tl.setYRange(-0.05, 6.05, padding=0)
        tl.getAxis("left").setTicks([[(i + 0.4, f"{string_number(i)} {OPEN_NAMES[i]}") for i in range(6)]])
        tl.getAxis("left").setWidth(AXIS_W)
        tl.setLabel("bottom", "time (s)")
        tl.setTitle("String activity (numbers = fret detected at each strike)", color="#94a3b8", size="10pt")
        for i in range(7):
            tl.addItem(pg.InfiniteLine(pos=i, angle=0, pen=pg.mkPen("#1f2937")))
        self.curves = []
        for i in range(6):
            fill = QColor(STRING_COLORS[i])
            fill.setAlpha(70)
            self.curves.append(tl.plot(pen=pg.mkPen(STRING_COLORS[i], width=1.5), fillLevel=i, brush=fill))
        return tl

    def _build_tab(self):
        tab = self.tab_plot = pg.PlotWidget()
        tab.setMouseEnabled(False, False)
        tab.hideButtons()
        tab.setXLink(self.timeline)
        tab.setYRange(-0.6, 5.6, padding=0)
        tab.getAxis("left").setTicks([[(i, OPEN_NAMES[i]) for i in range(6)]])
        tab.getAxis("left").setWidth(AXIS_W)
        tab.getAxis("bottom").setStyle(showValues=False)
        tab.setTitle("Tab (high e on top; blue = moved by the model or fingering after it first appeared)",
                     color="#94a3b8", size="10pt")
        for i in range(6):
            tab.addItem(pg.InfiniteLine(pos=i, angle=0, pen=pg.mkPen("#475569")))
        return tab

    def _build_spectrum(self):
        sp = pg.PlotWidget()
        sp.setMouseEnabled(False, False)
        sp.hideButtons()
        sp.setLogMode(x=True, y=False)
        sp.setXRange(np.log10(60), np.log10(5000), padding=0)
        sp.setYRange(-110, 0, padding=0)
        sp.setLabel("bottom", "frequency (Hz)")
        sp.getAxis("bottom").setTicks([[(np.log10(f), str(f)) for f in (82, 110, 165, 220, 330, 500, 1000, 2000, 5000)]])
        sp.setTitle("Spectrum (gray = open strings, dashed = detected fundamentals)", color="#94a3b8", size="10pt")
        self.spec_curve = sp.plot(pen=pg.mkPen("#38bdf8", width=1))
        for midi in (40, 45, 50, 55, 59, 64):
            sp.addItem(pg.InfiniteLine(
                pos=np.log10(float(midi_to_hz(midi))), angle=90,
                pen=pg.mkPen("#334155", width=1),
            ))
        self.f0_lines = []
        for i in range(6):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(STRING_COLORS[i], width=1.5, style=Qt.DashLine))
            line.hide()
            sp.addItem(line)
            self.f0_lines.append(line)
        return sp

    # ----------------------------------------------------------------- update
    def start(self):
        self.source.start()
        self.source_label.setText(self.source.label)
        self.timer.start(TICK_MS)

    def _tick(self):
        frame = self.analyzer.process(self.source.ring)
        now = frame.time
        self.fretboard.set_state(frame.strings, now)
        self.meter.setValue(int(np.clip(frame.input_db + 60.0, 0, 60)))

        self.hist_t.append(now)
        self.hist_lv.append([i + 0.8 * _norm_level(s.level_db) if s.active else i for i, s in enumerate(frame.strings)])
        t = np.fromiter(self.hist_t, float)
        levels = np.array(self.hist_lv)
        for i, curve in enumerate(self.curves):
            curve.setData(t, levels[:, i])
        self.timeline.setXRange(now - TIMELINE_S, now, padding=0)

        self.spec_curve.setData(frame.freqs, frame.spectrum_db)
        for s, line in zip(frame.strings, self.f0_lines):
            if s.active:
                line.setPos(np.log10(float(midi_to_hz(s.midi))))
                line.show()
            else:
                line.hide()

        while self.source.truth:
            truth = self.source.truth.popleft()
            self._log(f"{truth['time']:7.2f}s  SIM     {truth['label']}", "#64748b")
        for ev in frame.events:
            self._handle_event(ev, now)

        for sid in [k for k, (_, t0) in self.strike_items.items() if t0 < now - TIMELINE_S - 1]:
            self.timeline.removeItem(self.strike_items.pop(sid)[0])
        self._update_tab(now)
        self._update_sheet()

    def _update_tab(self, now):
        oldest = now - TIMELINE_S - 1
        for note in reversed(self.analyzer.tab.notes):
            if note.time < oldest:
                break
            shown = self.tab_items.get(note.id)
            if shown and shown[1] == note.version:
                continue
            if shown:
                for item in shown[0]:
                    self.tab_plot.removeItem(item)
            color = TAB_REVISED_COLOR if note.version else TAB_COLOR
            items = []
            for i, fret in note.frets.items():
                item = pg.TextItem(str(fret), color=color, anchor=(0.5, 0.5), fill="#111827")
                item.setFont(QFont("Consolas", 11, QFont.Bold))
                item.setPos(note.time, i)
                self.tab_plot.addItem(item)
                items.append(item)
            self.tab_items[note.id] = (items, note.version, note.time)
        for note_id in [k for k, (_, _, t0) in self.tab_items.items() if t0 < oldest]:
            for item in self.tab_items.pop(note_id)[0]:
                self.tab_plot.removeItem(item)

    def _update_sheet(self):
        """Engrave the same MusicXML as the PDF, in a worker so audio stays smooth."""
        if self.sheet_job is not None:
            if not self.sheet_job.done():
                return
            job, self.sheet_job = self.sheet_job, None
            try:
                self.sheet.set_png(job.result())
            except Exception as exc:
                self.sheet_pool = False
                self._log(f"Sheet music unavailable: {exc}", "#fbbf24")
        tab = self.analyzer.tab
        state = (len(tab.notes), sum(n.version for n in tab.notes[-64:]),
                 tuple(n.end is None for n in tab.notes[-2:]))
        if (self.sheet_pool is False or not tab.notes or state == self.sheet_shown
                or time.monotonic() - self.sheet_sent_at < SHEET_REFRESH_S):
            return
        if self.sheet_pool is None:
            self.sheet_pool = ProcessPoolExecutor(max_workers=1)
        self.sheet_job = self.sheet_pool.submit(_engrave_live, tab.musicxml(last_bars=SHEET_BARS))
        self.sheet_shown = state
        self.sheet_sent_at = time.monotonic()

    def _export_tab(self):
        tab = self.analyzer.tab
        if not tab.notes:
            self._log("Nothing to export yet: no notes played", "#fbbf24")
            return
        TAB_DIR.mkdir(exist_ok=True)
        default = TAB_DIR / f"tab_{datetime.now():%Y%m%d_%H%M%S}.txt"
        name, _ = QFileDialog.getSaveFileName(self, "Export tab", str(default), "ASCII tab (*.txt)")
        if not name:
            return
        text_path = Path(name).with_suffix(".txt")
        xml_path = text_path.with_suffix(".musicxml")
        text_path.write_text(tab.ascii(), encoding="utf-8")
        xml = tab.musicxml()
        xml_path.write_text(xml, encoding="utf-8")
        written = [text_path.name, xml_path.name]
        try:
            from .sheet import render_pdf
            render_pdf(xml, text_path.with_suffix(".pdf"), text_path.stem)
            written.append(text_path.with_suffix(".pdf").name)
        except Exception as exc:
            self._log(f"Sheet music PDF skipped: {exc}", "#fbbf24")
        self._log(f"Exported {len(tab.notes)} notes to {', '.join(written)} in {text_path.parent}", "#4ade80")

    def _handle_event(self, ev, now):
        latency_ms = (ev.detected_at - ev.time) * 1000
        self.chord_label.setText(self.analyzer.chord)
        self.detail_label.setText(f"{ev.label}    ·    shown {latency_ms:.0f} ms after the pluck")
        for sid in ev.dropped:
            dropped = self.strike_items.pop(sid, None)
            if dropped:
                self.timeline.removeItem(dropped[0])
        for i, fret in ev.frets.items():
            replaced = self.strike_items.pop(ev.strike_ids[i], None)
            if replaced:
                self.timeline.removeItem(replaced[0])
            item = pg.TextItem(str(fret), color=QUALITY_COLORS["ringing"], anchor=(0.5, 0.5))
            item.setFont(QFont("Consolas", 11, QFont.Bold))
            item.setPos(ev.time, i + 0.45)
            self.timeline.addItem(item)
            self.strike_items[ev.strike_ids[i]] = (item, ev.time)
        self._log(f"{ev.time:7.2f}s  DETECT  {ev.label}   ({latency_ms:.0f} ms)", "#e2e8f0")

    def _log(self, text, color):
        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        self.log.insertItem(0, item)
        while self.log.count() > 300:
            self.log.takeItem(self.log.count() - 1)

    def closeEvent(self, event):
        self.timer.stop()
        self.source.stop()
        if self.sheet_pool:
            self.sheet_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
