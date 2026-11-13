"""Import the Hugging Face guitar-fretboard-notes dataset in the recorder's clip format.

    python import_hf_dataset.py          # writes dataset_hf/labels.csv and dataset_hf/clips/

Source: https://huggingface.co/datasets/collegefishiesd/guitar-fretboard-notes (CC BY-SA 4.0).
Clips are resampled to 48 kHz and cut like the recorder's: from 0.1 s before the first pluck, 2.1 s
long. Many takes pluck the same note again later; that stays in the clip, since the label still holds.
"""
import argparse
import csv
import io
import os
import shutil
import tempfile
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
import soxr

from guitar_hmi.analysis import ANALYSIS_WINDOW_S, PRE_ONSET_S, OnsetDetector, detect_pitches
from guitar_hmi.guitar import OPEN_MIDI, tab_string
from guitar_hmi.recorder import FIELDS, PRE_ROLL_S
from guitar_hmi.sources import RingBuffer

DATA_URL = "https://huggingface.co/datasets/collegefishiesd/guitar-fretboard-notes/resolve/main/data"
SPLITS = ("train", "validation", "test")
SR = 48000
CLIP_S = 2.0
BLOCK = 1024
QC_OFFSETS_S = (0.0, 0.05, 0.1, 0.2)
STRING_COLUMNS = ("fret_E", "fret_A", "fret_D", "fret_G", "fret_B", "fret_e")


def download(split, folder):
    path = os.path.join(folder, f"{split}.parquet")
    print(f"Downloading {split} split ...")
    with requests.get(f"{DATA_URL}/{split}-00000-of-00001.parquet", stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path


def find_plucks(x, sr):
    """Onsets from the recorder's detector, run over x as a live stream. Background noise
    triggers quiet onsets, so only onsets at least half as loud as the loudest one count."""
    ring = RingBuffer(sr, seconds=len(x) / sr + 1)
    detector = OnsetDetector(sr)
    detector.scan(ring)
    onsets = []
    for i in range(0, len(x), BLOCK):
        ring.write(x[i:i + BLOCK])
        onsets += detector.scan(ring)
    window = int(ANALYSIS_WINDOW_S * sr)
    rms = [np.sqrt(np.mean(x[o:o + window] ** 2)) for o in onsets]
    return [o for o, r in zip(onsets, rms) if r >= 0.5 * max(rms, default=0)]


def pitch_check(clip, onset, midi, sr):
    """Pitches heard at the pluck, and whether the target (or its octave) is heard within
    200 ms of it. The detector is tuned for the piezo and often reads these recordings
    two octaves low right at the attack."""
    a, n = int(PRE_ONSET_S * sr), int(ANALYSIS_WINDOW_S * sr)
    starts = [max(onset - a, 0) + int(t * sr) for t in QC_OFFSETS_S]
    heard = [detect_pitches(clip[s:s + n], sr) for s in starts if s + n <= len(clip)]
    passed = any(h - midi in (0, 12, -12) for pitches in heard for h in pitches[:4])
    return (heard[0] if heard else []), passed


def convert(item):
    """One dataset item as (clip, CSV row), or a reason string if it can't be used."""
    string, fret, midi = item["string_number"], item["fret"], item["midi_number"]
    index = 6 - string
    if OPEN_MIDI[index] + fret != midi:
        return "string/fret disagrees with MIDI note"

    x, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32", always_2d=True)
    x = soxr.resample(x.mean(axis=1), sr, SR)
    plucks = find_plucks(x, SR)
    if not plucks:
        return "no pluck detected"
    onset = plucks[0]
    start = max(onset - int(PRE_ROLL_S * SR), 0)
    clip = x[start:onset + int(CLIP_S * SR)]
    clip /= max(1.0, float(np.abs(clip).max()))
    onset -= start

    heard, passed = pitch_check(clip, onset, midi, SR)
    source = item["source"]
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "file": f"clips/{source}/{source}_s{string}f{fret}.wav", "session": f"hf_{source}",
        "kind": "note", "tab": tab_string({index: fret}), "string": string, "fret": fret, "midi": midi,
        "quality": "clean", "technique": item["guitar_type"],
        "onset_s": round(onset / SR, 4), "duration_s": round(len(clip) / SR, 3), "sr": SR,
        "peak": round(float(np.abs(clip).max()), 4),
        "qc_pass": int(passed), "heard": " ".join(str(m) for m in heard),
    })
    for i, col in enumerate(STRING_COLUMNS):
        row[col] = fret if i == index else -1
    return clip, row


def main():
    ap = argparse.ArgumentParser(description="Import the Hugging Face guitar-fretboard-notes dataset")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_hf"))
    args = ap.parse_args()

    shutil.rmtree(os.path.join(args.out, "clips"), ignore_errors=True)
    rows, skipped = [], Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for split in SPLITS:
            for item in pq.read_table(download(split, tmp)).to_pylist():
                result = convert(item)
                if isinstance(result, str):
                    skipped[result] += 1
                    continue
                clip, row = result
                path = os.path.join(args.out, row["file"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sf.write(path, clip, SR, subtype="PCM_24")
                rows.append(row)

    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = Counter(r["session"] for r in rows)
    failed = Counter(r["session"] for r in rows if not r["qc_pass"])
    for session in sorted(total):
        print(f"  {session:<15} {total[session]:>3} clips, {failed[session]:>2} failed the pitch check")
    print(f"Wrote {len(rows)} clips to {args.out}")
    for reason, n in skipped.items():
        print(f"  skipped {n}: {reason}")


if __name__ == "__main__":
    main()
