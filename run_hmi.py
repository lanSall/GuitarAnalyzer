"""Launch the real-time guitar string monitor.

    python run_hmi.py                         # built-in simulated guitar
    python run_hmi.py --source live --list-devices
    python run_hmi.py --source live --device "UM2" --channel 1
    python run_hmi.py --source file --file recording.wav
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Real-time per-string guitar monitor")
    ap.add_argument("--source", choices=["sim", "live", "file"], default="sim")
    ap.add_argument("--device", help="input device index or (partial) name, for --source live")
    ap.add_argument("--channel", type=int, default=0, help="0-based input channel (UM2 instrument input is usually 1)")
    ap.add_argument("--sr", type=int, default=48000, help="sample rate for live input")
    ap.add_argument("--asio", action="store_true", help="enable ASIO devices (lowest latency on Windows)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--file", help="WAV/FLAC file for --source file")
    ap.add_argument("--seed", type=int, help="simulator random seed")
    ap.add_argument("--snapshot", help="save a screenshot to this path after --snapshot-after seconds, then exit")
    ap.add_argument("--snapshot-after", type=float, default=8.0)
    args = ap.parse_args()

    if args.asio:
        os.environ["SD_ENABLE_ASIO"] = "1"
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from guitar_hmi.analysis import DspAnalyzer
    from guitar_hmi.sources import FileSource, LiveSource, SimulatedSource
    from guitar_hmi.ui import MainWindow

    if args.source == "live":
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        source = LiveSource(args.sr, device, args.channel)
    elif args.source == "file":
        if not args.file:
            ap.error("--source file needs --file")
        source = FileSource(args.file, args.channel)
    else:
        source = SimulatedSource(48000, args.seed)

    app = QApplication(sys.argv)
    window = MainWindow(source, DspAnalyzer(source.sr))
    window.resize(1400, 920)
    window.show()
    window.start()
    if args.snapshot:
        QTimer.singleShot(int(args.snapshot_after * 1000), lambda: (window.grab().save(args.snapshot), app.quit()))
    code = app.exec()
    source.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()
