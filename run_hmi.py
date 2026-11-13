"""Launch the real-time guitar string monitor.

    python run_hmi.py                         # built-in simulated guitar
    python run_hmi.py --source live --list-devices
    python run_hmi.py --source live --device "UM2" --channel 1
    python run_hmi.py --source file --file recording.wav
    python run_hmi.py --model                 # let the trained model pick the string (models/string_model.pt)
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
    ap.add_argument("--json", action="store_true", help="with --list-devices: print input devices as JSON")
    ap.add_argument("--file", help="WAV/FLAC file for --source file")
    ap.add_argument("--seed", type=int, help="simulator random seed")
    ap.add_argument("--snapshot", help="save a screenshot to this path after --snapshot-after seconds, then exit")
    ap.add_argument("--snapshot-after", type=float, default=8.0)
    ap.add_argument("--model", nargs="?", const="", metavar="PATH",
                    help="use the trained string model (default models/string_model.pt)")
    args = ap.parse_args()

    if args.asio:
        os.environ["SD_ENABLE_ASIO"] = "1"
    if args.list_devices:
        import sounddevice as sd

        if args.json:
            import json

            apis = sd.query_hostapis()
            print(json.dumps([
                {"index": d["index"], "name": d["name"], "hostapi": apis[d["hostapi"]]["name"],
                 "channels": d["max_input_channels"]}
                for d in sd.query_devices() if d["max_input_channels"] > 0
            ]))
        else:
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

    if args.model is not None:
        from guitar_hmi.model_analyzer import DEFAULT_MODEL, ModelAnalyzer

        path = args.model or DEFAULT_MODEL
        if not os.path.exists(path):
            ap.error(f"no model at {path}; train one with: python Training\\Training.py --save")
        analyzer = ModelAnalyzer(source.sr, path)
    else:
        analyzer = DspAnalyzer(source.sr)

    app = QApplication(sys.argv)
    window = MainWindow(source, analyzer)
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
