"""Train the per-string chord model on strums mixed from single-note clips.

    python Training\\train_chords.py --save              train on every guitar, save models/chord_model.pt
    python Training\\train_chords.py --val-group hf_eqm  hold one guitar out and score strums mixed from it

Real labeled strums (kind "chord" in labels.csv) are never trained on; they are scored after
training when there are any. Training\\eval_chords.py compares this model with the template fitter.
"""
import argparse
import random
import time
from pathlib import Path

import torch

from chords import (StrumMixer, chord_rows, frets_of, load_chord_clips, predict_shapes, save_chord_model,
                    score_shapes, train_chord_model)
from data import ROOT, load_clips, load_rows

DEFAULT_MODEL = ROOT / "models" / "chord_model.pt"


def add_training_args(ap):
    ap.add_argument("--dataset", type=Path, nargs="+", default=[ROOT / "dataset", ROOT / "dataset_hf"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40, help="batches of freshly mixed strums per epoch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6, help="CPU threads for training (leave some for audio)")
    ap.add_argument("--test-strums", type=int, default=300)
    ap.add_argument("--include-electric", action="store_true", help="also mix strums from the electric HF guitars")


def load_notes(datasets, electric=False):
    rows = [r for d in datasets for r in load_rows(d, electric=electric)]
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, load_clips(rows)


def test_strums(mixer, group, n, seed):
    """n fixed strums mixed from one group (no augmentation), with their true shapes."""
    rng = random.Random(seed)
    audio, truth = [], []
    for _ in range(n):
        a, t = mixer.strum(group, mixer.shape(group, rng), rng)
        audio.append(a)
        truth.append(frets_of(t.tolist()))
    return torch.stack(audio), truth


def real_strums(datasets):
    rows = [r for d in datasets for r in chord_rows(d) if r["path"].exists()]
    return rows, load_chord_clips(rows)


def report(label, result):
    print(f"{label:<34} exact {result['exact']:6.1%}   per string {result['per_string']:6.1%}   "
          f"chord name {result['name']:6.1%}")


def main():
    ap = argparse.ArgumentParser(description="Train the per-string chord model on mixed strums")
    add_training_args(ap)
    ap.add_argument("--val-group", help="hold out this guitar group and score strums mixed from it")
    ap.add_argument("--save", type=Path, nargs="?", const=DEFAULT_MODEL,
                    help=f"train on every guitar and save the model (default {DEFAULT_MODEL})")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if not args.save and not args.val_group:
        ap.error("give --save or --val-group")

    rows, clips = load_notes(args.dataset, args.include_electric)
    groups = sorted({r["group"] for r in rows})
    train_groups = [g for g in groups if g != args.val_group]
    mixer = StrumMixer(rows, clips, set(groups))
    started = time.perf_counter()
    model = train_chord_model(mixer, train_groups, args.epochs, args.steps, args.batch_size, args.lr, args.seed)
    print(f"Trained on strums mixed from {', '.join(train_groups)} in {time.perf_counter() - started:.0f} s")

    if args.val_group:
        audio, truth = test_strums(mixer, args.val_group, args.test_strums, args.seed + 1)
        predicted = [frets_of(t.tolist()) for t in predict_shapes(model, audio)[0]]
        report(f"{args.val_group} (held out, mixed)", score_shapes(truth, predicted))
    real, real_clips = real_strums(args.dataset)
    if real:
        predicted = [frets_of(t.tolist()) for t in predict_shapes(model, real_clips)[0]]
        report(f"real strums ({len(real)})", score_shapes([r["frets"] for r in real], predicted))
    if args.save:
        save_chord_model(args.save, model)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
