"""Guitar Analyzer start-up window: pick the audio input once, launch any tool."""
import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from guitar_hmi.ui import STYLE

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL = ROOT / "models" / "string_model.pt"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def console_python():
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe")
    return str(candidate if exe.name.lower() == "pythonw.exe" and candidate.exists() else exe)


class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Analyzer")
        self.settings = QSettings("GuitarAnalyzer", "Launcher")
        self.python = console_python()
        self.procs = []
        self.setStyleSheet(STYLE + """
            QGroupBox { border: 1px solid #1f2937; border-radius: 6px; margin-top: 14px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #94a3b8; }
            QPushButton { background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: left; }
            QPushButton:hover { background: #334155; }
            QPushButton#small { padding: 4px 10px; }
            QComboBox, QSpinBox { background: #111827; border: 1px solid #334155; padding: 3px; }
            QPlainTextEdit { background: #111827; border: 1px solid #1f2937; font-family: Consolas; font-size: 11px; }
        """)

        central = QWidget()
        v = QVBoxLayout(central)
        title = QLabel("Guitar Analyzer")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        v.addWidget(title)
        sub = QLabel("Real-time string monitor and training-data tools for a piezo-equipped guitar")
        sub.setStyleSheet("color: #94a3b8;")
        v.addWidget(sub)
        v.addWidget(self._build_input())
        v.addWidget(self._build_tools())
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Tool output appears here")
        v.addWidget(self.log, 1)
        self.setCentralWidget(central)
        self.refresh_devices()

    # ----------------------------------------------------------------- layout
    def _build_input(self):
        box = QGroupBox("Audio input (used by the live monitor and the recorder)")
        row = QHBoxLayout(box)
        self.device = QComboBox()
        self.device.setMinimumWidth(420)
        row.addWidget(QLabel("Device"))
        row.addWidget(self.device, 1)
        refresh = QPushButton("Refresh")
        refresh.setObjectName("small")
        refresh.clicked.connect(self.refresh_devices)
        row.addWidget(refresh)
        row.addWidget(QLabel("Channel"))
        self.channel = QSpinBox()
        self.channel.setRange(1, 32)
        self.channel.setValue(int(self.settings.value("channel", 2)))
        self.channel.setToolTip("1-based input channel. On a Behringer UM2 the instrument (Hi-Z) input is channel 2.")
        row.addWidget(self.channel)
        self.asio = QCheckBox("ASIO (lowest latency)")
        self.asio.setChecked(self.settings.value("asio", "false") == "true")
        self.asio.toggled.connect(self.refresh_devices)
        row.addWidget(self.asio)
        self.use_model = QCheckBox("Trained string model")
        self.use_model.setToolTip("Monitors let the model trained by Training\\Training.py --save pick the string "
                                  f"for single notes, about 1.2 s after the pluck ({MODEL.relative_to(ROOT)}).")
        self.use_model.setChecked(self.settings.value("use_model", "false") == "true")
        row.addWidget(self.use_model)
        return box

    def _build_tools(self):
        box = QGroupBox("Tools")
        grid = QGridLayout(box)
        tools = [
            ("Live monitor", "See every string in real time from your guitar", self.live_monitor),
            ("Simulator monitor", "Same display, driven by a simulated guitar (no hardware needed)", self.sim_monitor),
            ("Record training data", "Prompted sessions that save labeled clips to the dataset", self.record_live),
            ("Try the recorder with the simulator", "A simulated player follows the prompts", self.record_sim),
            ("Play a recording…", "Run the monitor on a WAV file (e.g. a saved session)", self.play_file),
            ("Accuracy benchmark", "Score the analyzer against simulated ground truth", self.benchmark),
            ("Open dataset folder", "Clips, labels.csv and full session recordings", self.open_dataset),
            ("Create desktop shortcut", "Start Guitar Analyzer from your desktop", self.create_shortcut),
        ]
        for n, (name, desc, slot) in enumerate(tools):
            b = QPushButton(f"{name}\n{desc}")
            b.setFont(QFont("Segoe UI", 10))
            b.clicked.connect(slot)
            grid.addWidget(b, n // 2, n % 2)
        return box

    # ---------------------------------------------------------------- devices
    def refresh_devices(self):
        self.settings.setValue("asio", "true" if self.asio.isChecked() else "false")
        env = dict(os.environ, SD_ENABLE_ASIO="1") if self.asio.isChecked() else dict(os.environ)
        self.device.clear()
        self.device.addItem("System default input", None)
        try:
            out = subprocess.run([self.python, str(ROOT / "run_hmi.py"), "--list-devices", "--json"],
                                 capture_output=True, text=True, env=env, timeout=30, creationflags=NO_WINDOW)
            devices = json.loads(out.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            self._write(f"Could not list audio devices: {exc}")
            return
        saved = self.settings.value("device_name", "")
        saved_api = self.settings.value("device_hostapi", "")
        pick = 0
        best = (99, 99, 99)
        for d in devices:
            self.device.addItem(f"{d['name']}   ({d['hostapi']}, {d['channels']} ch)", d)
            idx = self.device.count() - 1
            name = d["name"].upper()
            api = d["hostapi"].upper()
            mapper = "MAPPER" in name or d["name"].lower().startswith("primary sound")
            interface = 0 if any(tag in name for tag in ("UM2", "U-PHORIA", "USB AUDIO CODEC", "BEHRINGER")) else 1
            api_rank = 0 if "WDM-KS" in api else 1 if "ASIO" in api else 2 if "WASAPI" in api else 3
            mic = 0 if "MICROPHONE" in name else 1
            if saved and d["name"] == saved and (not saved_api or d["hostapi"] == saved_api):
                score = (-1, api_rank, mic, 1 if mapper else 0)
            else:
                score = (interface, api_rank, mic, 1 if mapper else 0)
            if score < best:
                best = score
                pick = idx
        self.device.setCurrentIndex(pick)
        if devices:
            self._write(f"Found {len(devices)} input devices" + (" (ASIO enabled)" if self.asio.isChecked() else ""))
        else:
            self._write("No audio inputs found. Plug in the UM2 (or another interface) and press Refresh. "
                        "The simulator tools work without one.")

    def _input_args(self):
        d = self.device.currentData()
        self.settings.setValue("device_name", d["name"] if d else "")
        self.settings.setValue("device_hostapi", d["hostapi"] if d else "")
        self.settings.setValue("channel", self.channel.value())
        args = ["--source", "live", "--channel", str(self.channel.value() - 1)]
        if d:
            args += ["--device", str(d["index"])]
            if d["channels"] < self.channel.value():
                self._write(f"Warning: {d['name']} has only {d['channels']} input channel(s); "
                            f"channel {self.channel.value()} will fail.")
        if self.asio.isChecked():
            args.append("--asio")
        return args

    # ------------------------------------------------------------------ tools
    def _model_args(self):
        self.settings.setValue("use_model", "true" if self.use_model.isChecked() else "false")
        if not self.use_model.isChecked():
            return []
        if not MODEL.exists():
            self._write(f"No trained model at {MODEL}; run: python Training\\Training.py --save. "
                        "Using the DSP analyzer.")
            return []
        return ["--model", str(MODEL)]

    def live_monitor(self):
        self._run("Live monitor", "run_hmi.py", self._input_args() + self._model_args())

    def sim_monitor(self):
        self._run("Simulator monitor", "run_hmi.py", ["--source", "sim"] + self._model_args())

    def record_live(self):
        self._run("Recorder", "run_recorder.py", self._input_args() + ["--out", str(DATASET)])

    def record_sim(self):
        self._run("Recorder (sim)", "run_recorder.py", ["--source", "sim", "--out", str(ROOT / "dataset_sim")])

    def play_file(self):
        start = str(DATASET / "sessions") if (DATASET / "sessions").exists() else str(ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Choose a recording", start, "Audio (*.wav *.flac)")
        if path:
            self._run("File monitor", "run_hmi.py", ["--source", "file", "--file", path] + self._model_args())

    def benchmark(self):
        self._run("Benchmark", "sim_benchmark.py", ["--events", "100"])

    def open_dataset(self):
        DATASET.mkdir(exist_ok=True)
        os.startfile(DATASET)

    def create_shortcut(self):
        target = ROOT / "Guitar Analyzer.bat"
        script = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            "[Environment]::GetFolderPath('Desktop') + '\\Guitar Analyzer.lnk');"
            f"$s.TargetPath = '{target}'; $s.WorkingDirectory = '{ROOT}';"
            "$s.WindowStyle = 7; $s.IconLocation = 'imageres.dll,103'; $s.Save()"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                capture_output=True, text=True, creationflags=NO_WINDOW)
        self._write("Desktop shortcut created: Guitar Analyzer" if result.returncode == 0
                    else f"Could not create shortcut: {result.stderr.strip()}")

    # ------------------------------------------------------------- processes
    def _run(self, name, script, args):
        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._write(bytes(proc.readAllStandardOutput()).decode(errors="replace").rstrip(), name))
        proc.finished.connect(lambda code, _status: self._finished(proc, name, code))
        self._write(f"Starting {name}…")
        proc.start(self.python, [str(ROOT / script), *args])
        self.procs.append(proc)

    def _finished(self, proc, name, code):
        self._write(f"{name} closed" + ("" if code == 0 else f" with an error (exit code {code})"))
        if proc in self.procs:
            self.procs.remove(proc)

    def _write(self, text, name=None):
        if text:
            prefix = f"[{name}] " if name else ""
            self.log.appendPlainText("\n".join(prefix + line for line in text.splitlines()))

    def closeEvent(self, event):
        for proc in self.procs:
            proc.kill()
        super().closeEvent(event)


def main():
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    window = Launcher()
    window.resize(980, 760)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
