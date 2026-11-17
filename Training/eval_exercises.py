"""Score the monitor on notes plucked while earlier notes still ring.

    python Training\\eval_exercises.py                  recorded let-ring exercises (record_exercise.py)
    python Training\\eval_exercises.py --synthetic 60   random let-ring sequences mixed from your clips
    python Training\\eval_exercises.py --analyzer dsp   the DSP baseline without the string model

Each take is replayed through the analyzer in picking mode; the tab it writes is lined up with the
expected notes in order, and every expected note is scored by what was ringing when it was plucked.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from data import take_half

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from guitar_hmi.analysis import DspAnalyzer
from guitar_hmi.exercises import midi_of
from guitar_hmi.sources import RingBuffer

BLOCK_S = 0.016
TAIL_S = 0.6  # silence appended so decisions due after the last pluck still happen
# Semitone gaps whose overtones coincide (unison, fifth, octave, octave + fifth, two octaves).
SHARED_OVERTONES = {0, 7, 12, 19, 24}
CONTEXTS = ["nothing ringing", "over ringing", "over ringing, shared overtones"]


def recorded_takes(folder, take_numbers=None):
    takes = []
    for meta in sorted(Path(folder).rglob("*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        if take_numbers and info["take"] not in take_numbers:
            continue
        audio, sr = sf.read(meta.with_suffix(".wav"), dtype="float32", always_2d=True)
        takes.append((f"{meta.parent.name}/{meta.stem}", audio[:, 0], sr, [tuple(n) for n in info["notes"]], None))
    return takes


def synthetic_takes(n, seed, sessions=None, half=None, notes_per_take=8, max_fret=5):
    """Let-ring sequences mixed from clean single-note clips of your guitar: each pluck lands
    0.45-1.0 s after the last, earlier notes keep ringing, and replucking a string cuts it."""
    rows = [r for r in csv.DictReader(open(ROOT / "dataset" / "labels.csv", encoding="utf-8"))
            if r["kind"] == "note" and r["quality"] == "clean" and int(r["fret"]) <= max_fret
            and (not sessions or r["session"] in sessions)
            and (half is None or take_half(r["file"]) == half)]
    by_pos = {}
    for r in rows:
        by_pos.setdefault((int(r["string"]), int(r["fret"])), []).append(r)
    positions = sorted(by_pos)
    rng = np.random.default_rng(seed)
    cache = {}

    def clip(row):
        if row["file"] not in cache:
            a, sr = sf.read(ROOT / "dataset" / row["file"], dtype="float32", always_2d=True)
            a = a[:, 0] / (np.max(np.abs(a[:, 0])) + 1e-9)
            cache[row["file"]] = (a[int(float(row["onset_s"]) * sr):], sr)
        return cache[row["file"]]

    takes = []
    for t in range(n):
        notes = [positions[k] for k in rng.choice(len(positions), notes_per_take)]
        onsets = 0.3 + np.concatenate([[0.0], np.cumsum(rng.uniform(0.45, 1.0, notes_per_take - 1))])
        parts = []
        for (string, fret), start in zip(notes, onsets):
            row = by_pos[(string, fret)][rng.integers(len(by_pos[(string, fret)]))]
            a, sr = clip(row)
            parts.append((string, start, a * 10 ** (rng.uniform(-6, 0) / 20)))
        out = np.zeros(int((onsets[-1] + 2.2) * sr), np.float32)
        fade = int(0.01 * sr)
        for k, (string, start, a) in enumerate(parts):
            later = [s for st, s, _ in parts[k + 1:] if st == string]
            a = a[:int((later[0] - start) * sr) + fade].copy() if later else a.copy()
            # Ending a clip abruptly clicks like a pluck; real strings fade.
            a[-(fade if later else 10 * fade):] *= np.linspace(1, 0, fade if later else 10 * fade)
            i = int(start * sr)
            out[i:i + len(a)] += a[:len(out) - i]
        takes.append((f"synthetic {t + 1}", out, sr, notes, onsets))
    return takes


def replay_notes(audio, sr, analyzer):
    """(pluck time in seconds, (string number, fret)) of every note the analyzer tabbed, in order."""
    ring = RingBuffer(sr)
    block = int(BLOCK_S * sr)
    audio = np.concatenate([audio, np.zeros(int(TAIL_S * sr), np.float32)])
    for i in range(0, len(audio), block):
        ring.write(audio[i:i + block])
        analyzer.process(ring)
    found = []
    for note in analyzer.tab.notes:
        for index, fret in sorted(note.frets.items()):
            found.append((note.time, (6 - index, fret)))
    return found


def replay(audio, sr, analyzer):
    """(string number, fret) of every note the analyzer tabbed, in order."""
    return [n for _, n in replay_notes(audio, sr, analyzer)]


def align(expected, found):
    """Pairs (i expected, j found or None) from an edit-distance alignment on pitch."""
    a = [midi_of(*n) for n in expected]
    b = [midi_of(*n) for n in found]
    cost = np.zeros((len(a) + 1, len(b) + 1))
    cost[:, 0] = np.arange(len(a) + 1)
    cost[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]), cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    pairs, i, j = [], len(a), len(b)
    while i > 0:
        if j > 0 and cost[i, j] == cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and cost[i, j] == cost[i, j - 1] + 1:
            j -= 1
        else:
            pairs.append((i - 1, None))
            i -= 1
    return pairs[::-1], len(b) - sum(j is not None for _, j in pairs)


def ringing_before(expected, k):
    """Latest earlier note on each other string; plucking a string stops what it was playing."""
    latest = {}
    for string, fret in expected[:k]:
        latest[string] = (string, fret)
    latest.pop(expected[k][0], None)
    return list(latest.values())


def context(expected, k):
    ringing = ringing_before(expected, k)
    if not ringing:
        return CONTEXTS[0], ringing
    midi = midi_of(*expected[k])
    shared = any(abs(midi - midi_of(*n)) in SHARED_OVERTONES for n in ringing)
    return CONTEXTS[2 if shared else 1], ringing


def make_analyzer(kind, sr, model):
    if kind == "dsp":
        analyzer = DspAnalyzer(sr)
    else:
        from guitar_hmi.model_analyzer import ModelAnalyzer
        analyzer = ModelAnalyzer(sr, model_path=model, templates_path=None, chord_model_path=None)
    analyzer.mode = "picking"
    return analyzer


def main():
    ap = argparse.ArgumentParser(description="Score notes plucked over ringing strings")
    ap.add_argument("--dir", default=str(ROOT / "dataset" / "exercises"), help="recorded exercises")
    ap.add_argument("--takes", type=int, nargs="+",
                    help="only these take numbers (the ones exercise_clips.py --takes left out of training)")
    ap.add_argument("--synthetic", type=int, default=0, help="score this many mixed sequences instead")
    ap.add_argument("--sessions", nargs="+", help="mix synthetic sequences only from these recording sessions")
    ap.add_argument("--half", type=int, choices=[0, 1],
                    help="mix synthetic sequences only from this half of the takes (see Training.py --exclude-half)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyzer", choices=["model", "dsp"], default="model")
    ap.add_argument("--model", default=str(ROOT / "models" / "string_model.pt"))
    ap.add_argument("--verbose", action="store_true", help="print every take's mistakes")
    args = ap.parse_args()

    takes = (synthetic_takes(args.synthetic, args.seed, args.sessions, args.half) if args.synthetic
             else recorded_takes(args.dir, args.takes))
    if not takes:
        sys.exit(f"No exercises in {args.dir}; record some with record_exercise.py or use --synthetic N")

    stats = {c: Counter() for c in CONTEXTS}
    extra = 0
    echoed = 0
    for name, audio, sr, expected, _ in takes:
        found = replay(audio, sr, make_analyzer(args.analyzer, sr, args.model))
        pairs, n_extra = align(expected, found)
        extra += n_extra
        mistakes = []
        for i, j in pairs:
            ctx, ringing = context(expected, i)
            s = stats[ctx]
            s["notes"] += 1
            want = expected[i]
            got = found[j] if j is not None else None
            if got is None:
                mistakes.append(f"s{want[0]}f{want[1]} missed")
                continue
            s["found"] += 1
            if midi_of(*got) == midi_of(*want):
                s["pitch"] += 1
                s["position"] += got == want
                if got != want:
                    mistakes.append(f"s{want[0]}f{want[1]} placed on s{got[0]}f{got[1]}")
            else:
                repeat = any(midi_of(*got) == midi_of(*n) for n in ringing)
                echoed += repeat
                mistakes.append(f"s{want[0]}f{want[1]} heard as s{got[0]}f{got[1]}"
                                + ("  (a note still ringing)" if repeat else ""))
        if args.verbose:
            print(f"{name}: {len(expected)} expected, {len(found)} tabbed"
                  + (f"; {'; '.join(mistakes)}" if mistakes else "; all correct"))

    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "    -"

    print(f"\n{len(takes)} takes, analyzer: {args.analyzer}"
          + (f", model {Path(args.model).name}" if args.analyzer == "model" else ""))
    print(f"{'':32} {'notes':>5}  {'found':>6}  {'pitch':>6}  {'string+fret':>11}  {'placed if pitch ok':>18}")
    total = Counter()
    for c in CONTEXTS:
        s = stats[c]
        total.update(s)
        print(f"{c:32} {s['notes']:5d}  {pct(s['found'], s['notes'])}  {pct(s['pitch'], s['notes'])}  "
              f"{pct(s['position'], s['notes']):>11}  {pct(s['position'], s['pitch']):>18}")
    print(f"{'all notes':32} {total['notes']:5d}  {pct(total['found'], total['notes'])}  "
          f"{pct(total['pitch'], total['notes'])}  {pct(total['position'], total['notes']):>11}  "
          f"{pct(total['position'], total['pitch']):>18}")
    print(f"extra notes tabbed: {extra}; wrong pitches that were a note still ringing: {echoed}")


if __name__ == "__main__":
    main()
