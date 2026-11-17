"""Record the let-ring exercises (guitar_hmi/exercises.py) for scoring notes over ringing strings.

    python record_exercise.py --device 19 --channel 1
    python record_exercise.py --device 19 --channel 1 --only octaves --takes 2

Close the monitor first; the UM2 input usually can't be shared. Each take is saved as
dataset/exercises/<session>/<exercise>_t<take>.wav with a .json holding the expected notes.
Score them with: python Training\\eval_exercises.py
"""
import argparse
import json
import os
import threading
from datetime import datetime

import numpy as np
import soundfile as sf

from guitar_hmi.analysis import FluxOnsetDetector
from guitar_hmi.exercises import EXERCISES, ascii_tab
from guitar_hmi.sources import LiveSource, RingBuffer

ROOT = os.path.dirname(os.path.abspath(__file__))
# Farther apart than this and the earlier strings have faded, so the take no longer tests
# notes over ringing strings.
MAX_GAP_S = 2.0


class Tape:
    """Stands in for LiveSource's ring buffer and keeps every block while armed."""

    def __init__(self):
        self.armed = False
        self.blocks = []
        self.lock = threading.Lock()

    def write(self, x):
        if self.armed:
            with self.lock:
                self.blocks.append(np.array(x, dtype=np.float32))

    def take(self):
        with self.lock:
            audio = np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)
            self.blocks = []
        return audio


def find_plucks(audio, sr):
    """Pluck times in seconds, found the way the monitor finds them."""
    ring = RingBuffer(sr)
    detector = FluxOnsetDetector(sr)
    block = int(0.016 * sr)
    found = []
    for i in range(0, len(audio), block):
        ring.write(audio[i:i + block])
        found += detector.scan(ring)
    return [f / sr for f in found]


def main():
    ap = argparse.ArgumentParser(description="Record let-ring exercises with known notes")
    ap.add_argument("--device", help="input device index or (partial) name")
    ap.add_argument("--channel", type=int, default=0, help="0-based input channel (UM2 instrument input is usually 1)")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--takes", type=int, default=3, help="takes of each exercise")
    ap.add_argument("--only", nargs="+", choices=sorted(EXERCISES), help="record just these exercises")
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "exercises"))
    args = ap.parse_args()

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    source = LiveSource(args.sr, device, args.channel)
    tape = source.ring = Tape()
    source.start()
    print(f"Input: {source.label}")
    if "UM2" not in source.label.upper() and "USB AUDIO CODEC" not in source.label.upper():
        print("  Warning: this doesn't look like the UM2. Close the monitor and check --device.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, session)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    try:
        for name in args.only or EXERCISES:
            description, notes = EXERCISES[name]
            take = 1
            while take <= args.takes:
                print(f"\n=== {name}, take {take} of {args.takes}: {description}")
                print(ascii_tab(notes))
                answer = input("Enter to start (s = skip this exercise, q = quit): ").strip().lower()
                if answer in ("s", "q"):
                    break
                tape.take()
                tape.armed = True
                input("Recording. About one note per second, letting every note ring over the next; "
                      "Enter when the last note has faded: ")
                tape.armed = False
                audio = tape.take()
                peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-9) if len(audio) else -120.0
                plucks = find_plucks(audio, args.sr)
                heard = len(plucks)
                print(f"  {len(audio) / args.sr:.1f} s, peak {peak:.0f} dBFS, heard {heard} plucks "
                      f"(the exercise has {len(notes)})")
                gap = float(np.median(np.diff(plucks))) if heard > 1 else 0.0
                if gap > MAX_GAP_S:
                    print(f"  Notes were {gap:.1f} s apart, so earlier strings had faded. "
                          f"Redo it at about one note per second.")
                if input("  Enter to keep, r to redo: ").strip().lower() == "r":
                    continue
                stem = os.path.join(out_dir, f"{name}_t{take}")
                sf.write(stem + ".wav", audio, args.sr, subtype="FLOAT")
                with open(stem + ".json", "w", encoding="utf-8") as f:
                    json.dump(dict(exercise=name, take=take, notes=notes, sr=args.sr, input=source.label,
                                   recorded_at=datetime.now().isoformat(timespec="seconds"),
                                   plucks_heard=heard), f, indent=1)
                saved += 1
                take += 1
            if answer == "q":
                break
    finally:
        source.stop()
    print(f"\nSaved {saved} takes to {out_dir}")


if __name__ == "__main__":
    main()
