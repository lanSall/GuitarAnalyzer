"""Cut recorded let-ring exercises into labeled training clips of notes plucked over ringing strings.

    python Training\\exercise_clips.py                       every take in dataset/exercises and exercises_slow
    python Training\\exercise_clips.py --takes 1 2           only takes 1 and 2 (score take 3 with eval_exercises.py --takes 3)

Each take is replayed through the DSP analyzer; its notes are lined up with the exercise's notes
by pitch, and every note heard at the right pitch becomes a clip labeled with the exercise's
string and fret. Notes heard at the wrong pitch are skipped rather than guessed. The clips go to
dataset_ringing/, which Training.py reads along with the other datasets, so rerun this and
retrain whenever exercises are added. Never score a model on takes it was trained on.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from data import ONSET_S
from eval_exercises import align, replay_notes

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from guitar_hmi.analysis import DspAnalyzer
from guitar_hmi.exercises import midi_of

CLIP_S = 0.6
FIELDS = ["file", "session", "recorded_at", "kind", "quality", "string", "fret", "midi",
          "onset_s", "sr", "qc_pass", "exercise", "take"]


def main():
    ap = argparse.ArgumentParser(description="Cut let-ring exercise takes into labeled training clips")
    ap.add_argument("--dir", type=Path, nargs="+",
                    default=[ROOT / "dataset" / "exercises", ROOT / "dataset" / "exercises_slow"])
    ap.add_argument("--takes", type=int, nargs="+", help="only these take numbers")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset_ringing")
    args = ap.parse_args()

    shutil.rmtree(args.out / "clips", ignore_errors=True)
    rows, skipped = [], 0
    for meta in sorted(m for d in args.dir for m in d.rglob("*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        if args.takes and info["take"] not in args.takes:
            continue
        audio, sr = sf.read(meta.with_suffix(".wav"), dtype="float32", always_2d=True)
        audio = audio[:, 0]
        expected = [tuple(n) for n in info["notes"]]
        analyzer = DspAnalyzer(sr)
        analyzer.mode = "picking"
        found = replay_notes(audio, sr, analyzer)
        pairs, _ = align(expected, [n for _, n in found])
        session = f"ring_{meta.parent.name}"
        folder = args.out / "clips" / session
        folder.mkdir(parents=True, exist_ok=True)
        for i, j in pairs:
            string, fret = expected[i]
            if j is None or midi_of(*found[j][1]) != midi_of(string, fret):
                skipped += 1
                continue
            start = int((found[j][0] - ONSET_S) * sr)
            clip = audio[max(start, 0):start + int(CLIP_S * sr)]
            name = f"{meta.stem}_{i + 1:02d}_s{string}f{fret}.wav"
            sf.write(folder / name, clip, sr, subtype="PCM_24")
            rows.append({
                "file": f"clips/{session}/{name}", "session": session, "recorded_at": info.get("recorded_at", ""),
                "kind": "note", "quality": "clean", "string": string, "fret": fret,
                "midi": midi_of(string, fret), "onset_s": round(ONSET_S + min(start, 0) / sr, 4), "sr": sr,
                "qc_pass": 1, "exercise": info["exercise"], "take": info["take"],
            })
    with open(args.out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips of notes over ringing strings written to {args.out} "
          f"({skipped} notes skipped: heard at the wrong pitch or missed)")


if __name__ == "__main__":
    main()
