"""Score an analyzer against the simulator's ground truth (runs faster than real time).

    python sim_benchmark.py --events 200 --seed 1
"""
import argparse
import time

import numpy as np

from guitar_hmi.analysis import DspAnalyzer
from guitar_hmi.guitar import OPEN_MIDI
from guitar_hmi.sources import GuitarSynth, RingBuffer


def run(n_events, seed, sr=48000, tick_s=0.016, verbose=False):
    synth = GuitarSynth(sr, seed, mistake_rate=0.0)
    ring = RingBuffer(sr)
    analyzer = DspAnalyzer(sr)
    rng = np.random.default_rng(seed)
    tick = int(tick_s * sr)
    next_event = sr
    detections = []
    t0 = time.perf_counter()
    while len(synth.truth) < n_events or ring.total < next_event + sr:
        if ring.total >= next_event and len(synth.truth) < n_events:
            synth.trigger(ring.total)
            next_event = ring.total + int(rng.uniform(1.4, 2.6) * sr)
        ring.write(synth.render(tick))
        detections += analyzer.process(ring).events
    elapsed = time.perf_counter() - t0

    stats = {k: [0, 0, 0] for k in ("note", "chord")}  # [count, pitch ok, string+fret ok]
    latency, matched = [], set()
    for truth in synth.truth:
        kind = "chord" if len(truth["frets"]) > 1 else "note"
        stats[kind][0] += 1
        cands = [i for i, d in enumerate(detections) if -0.03 <= d.time - truth["time"] <= 0.1 and i not in matched]
        want_pitch = sorted(OPEN_MIDI[s] + f for s, f in truth["frets"].items())
        if cands:
            i = cands[0]
            matched.add(i)
            d = detections[i]
            got_pitch = sorted(OPEN_MIDI[s] + f for s, f in d.frets.items())
            stats[kind][1] += got_pitch == want_pitch
            stats[kind][2] += d.frets == truth["frets"]
            latency.append(d.detected_at - truth["time"])
            if verbose and d.frets != truth["frets"]:
                print(f"  truth {truth['label']:<40} got {d.label}")
        elif verbose:
            print(f"  MISSED {truth['label']}")

    audio_s = ring.total / sr
    print(f"{len(synth.truth)} events, {audio_s:.0f} s of audio analysed in {elapsed:.1f} s "
          f"({audio_s / elapsed:.1f}x real time)")
    for kind, (n, pitch_ok, exact_ok) in stats.items():
        if n:
            print(f"  {kind:<6} n={n:<4} pitch correct {pitch_ok / n:6.1%}   string+fret correct {exact_ok / n:6.1%}")
    print(f"  false detections: {len(detections) - len(matched)}")
    if latency:
        print(f"  latency onset->detection: mean {np.mean(latency) * 1000:.0f} ms, max {np.max(latency) * 1000:.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    run(args.events, args.seed, verbose=args.verbose)
