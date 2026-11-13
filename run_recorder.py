"""Launch the prompted training-data recorder.

    python run_recorder.py --source live --device "UM2" --channel 1
    python run_recorder.py --source sim          # a simulated player follows the prompts
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Record labeled training clips")
    ap.add_argument("--source", choices=["live", "sim"], default="live")
    ap.add_argument("--device", help="input device index or (partial) name")
    ap.add_argument("--channel", type=int, default=0, help="0-based input channel (UM2 instrument input is usually 1)")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--asio", action="store_true", help="enable ASIO devices")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
    args = ap.parse_args()

    if args.asio:
        os.environ["SD_ENABLE_ASIO"] = "1"

    from PySide6.QtWidgets import QApplication

    from guitar_hmi.recorder import RecorderWindow
    from guitar_hmi.sources import LiveSource, PromptedSimSource

    if args.source == "sim":
        source = PromptedSimSource(48000)
    else:
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        source = LiveSource(args.sr, device, args.channel)

    app = QApplication(sys.argv)
    window = RecorderWindow(source, args.out)
    window.resize(1400, 900)
    window.show()
    window.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
