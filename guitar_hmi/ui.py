"""Real-time HMI: fretboard, per-string timeline, spectrum and event log."""
import math
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from .guitar import NUM_FRETS, OPEN_NAMES, midi_to_hz, note_name, string_number

QUALITY_COLORS = {"ringing": "#4ade80", "muted": "#f87171", "buzz": "#fb923c", "idle": "#64748b"}
QUALITY_TEXT = {"ringing": "ringing", "muted": "MUTED", "buzz": "BUZZ", "idle": "idle"}
STRING_COLORS = ["#ef4444", "#f59e0b", "#eab308", "#22c55e", "#3b82f6", "#a855f7"]
GAUGES = [3.4, 2.8, 2.3, 1.8, 1.4, 1.1]
TIMELINE_S = 8.0
TICK_MS = 16

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
                   QUALITY_TEXT[st.quality if active else "idle"])
        bar_w = self.PANEL_W - 24
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1f2937"))
        p.drawRect(QRectF(px, y + 2, bar_w, 5))
        if active:
            p.setBrush(color)
            p.drawRect(QRectF(px, y + 2, bar_w * _norm_level(st.level_db), 5))


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
        layout.addWidget(self.fretboard, 4)
        layout.addWidget(self._build_timeline(), 3)
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_spectrum(), 1)
        self.log = QListWidget()
        bottom.addWidget(self.log, 1)
        layout.addLayout(bottom, 2)
        self.setCentralWidget(central)

        self.hist_t = deque(maxlen=int(TIMELINE_S * 1000 / TICK_MS) + 60)
        self.hist_lv = deque(maxlen=self.hist_t.maxlen)
        self.strike_items = {}
        self.mistake_until = 0.0
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
        self.detail_label = QLabel("Waiting for a strum…")
        self.detail_label.setFont(QFont("Segoe UI", 13))
        self.mistake_label = QLabel("")
        self.mistake_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        mid.addWidget(self.detail_label)
        mid.addWidget(self.mistake_label)
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
        header.addLayout(right)
        return header

    def _build_timeline(self):
        tl = self.timeline = pg.PlotWidget()
        tl.setMouseEnabled(False, False)
        tl.hideButtons()
        tl.setYRange(-0.05, 6.05, padding=0)
        tl.getAxis("left").setTicks([[(i + 0.4, f"{string_number(i)} {OPEN_NAMES[i]}") for i in range(6)]])
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

    def _build_spectrum(self):
        sp = pg.PlotWidget()
        sp.setMouseEnabled(False, False)
        sp.hideButtons()
        sp.setLogMode(x=True, y=False)
        sp.setXRange(np.log10(60), np.log10(5000), padding=0)
        sp.setYRange(-110, 0, padding=0)
        sp.setLabel("bottom", "frequency (Hz)")
        sp.getAxis("bottom").setTicks([[(np.log10(f), str(f)) for f in (82, 110, 165, 220, 330, 500, 1000, 2000, 5000)]])
        sp.setTitle("Spectrum (dashed = fundamentals of ringing strings)", color="#94a3b8", size="10pt")
        self.spec_curve = sp.plot(pen=pg.mkPen("#38bdf8", width=1))
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

        if self.mistake_until and now > self.mistake_until:
            self.mistake_label.setText("")
            self.mistake_until = 0.0
        for sid in [k for k, (_, t0) in self.strike_items.items() if t0 < now - TIMELINE_S - 1]:
            self.timeline.removeItem(self.strike_items.pop(sid)[0])

    def _handle_event(self, ev, now):
        latency_ms = (ev.detected_at - ev.time) * 1000
        if ev.kind == "mistake":
            color = QUALITY_COLORS[ev.quality]
            self.mistake_label.setText(f"⚠ {ev.label}")
            self.mistake_label.setStyleSheet(f"color: {color};")
            self.mistake_until = now + 3.0
            unsure = len(ev.strike_ids) > 1
            for i, sid in ev.strike_ids.items():
                if sid in self.strike_items:
                    item = self.strike_items[sid][0]
                    item.setColor(color)
                    if unsure:
                        item.setText(f"{ev.frets[i]}?")
            self._log(f"{ev.time:7.2f}s  MISTAKE {ev.label}", color)
            return

        self.chord_label.setText(self.analyzer.chord)
        self.detail_label.setText(f"{ev.label}    ·    shown {latency_ms:.0f} ms after the pluck")
        for i, fret in ev.frets.items():
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
        super().closeEvent(event)
