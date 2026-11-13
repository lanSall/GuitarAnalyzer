"""Train and evaluate a string/fret classifier on labeled guitar clips.

    python Training\\Training.py                        cross-validate, holding out one guitar at a time
    python Training\\Training.py --test                 train on every development guitar, score hf_deb once
    python Training\\Training.py --val-sessions hf_ele  one custom holdout
    python Training\\Training.py --save                 train on every clip and save the model for the monitor

Labels use the CSV's numbering: string 1 = high E ... 6 = low E, fret 0 = open. Each Hugging
Face source is one guitar group and all piezo sessions are another. hf_deb, the only
recordings by a second player, is the reserved test guitar: it is never trained on, and only
--test scores it. Every score is from the final epoch, so no result is picked by peeking.
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from augment import augment, spec_augment
from data import ROOT, TEST_GROUP, load_clips, load_rows, midi_of, split_by_groups
from features import N_HARMONIC_FEATURES, harmonic_stats, log_filterbank, spectrogram, standardized_harmonics
from model import Fretboard, NoteNet

METRICS = ("exact", "pitch", "string", "string_given_pitch")
HEADINGS = {"exact": "exact", "pitch": "pitch", "string": "string", "string_given_pitch": "string if pitch ok"}
EVAL_BATCH = 64
DEFAULT_MODEL = ROOT / "models" / "string_model.pt"


def train_fold(rows, clips, bank, args, seed):
    torch.manual_seed(seed)
    board = Fretboard({(r["string"], r["fret"]) for r in rows})
    x = clips[[r["idx"] for r in rows]]
    midi = torch.tensor([midi_of(r["string"], r["fret"]) for r in rows])
    pitch = torch.tensor([board.pitch_ids[m] for m in midi.tolist()])
    string = torch.tensor([r["string"] - 1 for r in rows])
    stats = harmonic_stats(x, midi) if args.harmonics else None

    model = NoteNet(bank.shape[0], len(board.pitches), N_HARMONIC_FEATURES if args.harmonics else 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * (len(rows) // args.batch_size)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(rows))
        for i in range(0, len(rows) - args.batch_size + 1, args.batch_size):
            b = order[i:i + args.batch_size]
            audio = augment(x[b]) if args.augment else x[b]
            spec = spectrogram(audio, bank)
            if args.augment:
                spec = spec_augment(spec)
            embedding = model.embed(spec)
            harmonics = standardized_harmonics(audio, midi[b], stats)
            loss = (loss_fn(model.pitch_head(embedding), pitch[b])
                    + loss_fn(model.string_logits(embedding, harmonics), string[b]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
    return model, board, stats


def predict(model, board, stats, clips, bank):
    """(string, fret) per clip. Pitch comes first; the harmonic features are then measured
    at the predicted pitch, as they would be in live use."""
    model.eval()
    predicted = []
    with torch.no_grad():
        for audio in clips.split(EVAL_BATCH):
            embedding = model.embed(spectrogram(audio, bank))
            pitch_ids = model.pitch_head(embedding).argmax(dim=1)
            harmonics = standardized_harmonics(audio, board.pitch_midi(pitch_ids), stats)
            predicted += board.decode(pitch_ids, model.string_logits(embedding, harmonics))
    return predicted


def save_model(path, model, board, stats, bank):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "positions": [list(p) for p in board.positions],
        "harmonic_mean": stats[0] if stats else None,
        "harmonic_std": stats[1] if stats else None,
        "n_bands": bank.shape[0],
    }, path)


def score(rows, predicted):
    truth = [(r["string"], r["fret"]) for r in rows]
    pitch_ok = [midi_of(*t) == midi_of(*p) for t, p in zip(truth, predicted)]
    string_ok = [t[0] == p[0] for t, p in zip(truth, predicted)]
    confusion = torch.zeros(6, 6, dtype=torch.long)
    for t, p in zip(truth, predicted):
        confusion[t[0] - 1, p[0] - 1] += 1
    n = len(truth)
    return {
        "exact": sum(t == p for t, p in zip(truth, predicted)) / n,
        "pitch": sum(pitch_ok) / n,
        "string": sum(string_ok) / n,
        "string_given_pitch": sum(s and p for s, p in zip(string_ok, pitch_ok)) / max(sum(pitch_ok), 1),
        "confusion": confusion,
    }


def spread(values):
    mean = statistics.mean(values)
    return f"{mean:6.1%}" + (f" +/- {statistics.stdev(values) * 100:4.1f}" if len(values) > 1 else "")


def report(results, confusion, seeds):
    print("\nfold            " + "".join(f"{HEADINGS[m]:>20}" for m in METRICS))
    for fold, runs in results.items():
        print(f"{fold:<16}" + "".join(f"{spread([r[m] for r in runs]):>20}" for m in METRICS))
    per_seed = [{m: statistics.mean(runs[k][m] for runs in results.values()) for m in METRICS}
                for k in range(len(seeds))]
    print(f"{'mean of folds':<16}" + "".join(f"{spread([s[m] for s in per_seed]):>20}" for m in METRICS))
    if len(seeds) > 1:
        print(f"(+/- is the standard deviation over seeds {', '.join(map(str, seeds))})")

    print("\nString confusion, all folds and seeds (rows = true string, columns = predicted)")
    print("       " + "".join(f"{f's{s}':>6}" for s in range(1, 7)))
    for s in range(6):
        print(f"  s{s + 1}   " + "".join(f"{int(v):>6}" for v in confusion[s]))


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate a string/fret classifier on labeled guitar clips")
    ap.add_argument("--dataset", type=Path, nargs="+", default=[ROOT / "dataset", ROOT / "dataset_hf"],
                    help="one or more dataset folders, each with a labels.csv")
    ap.add_argument("--val-sessions", nargs="+", help="hold out these groups or sessions instead of cross-validating")
    ap.add_argument("--test", action="store_true", help=f"train on every development guitar and score {TEST_GROUP}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--no-harmonics", dest="harmonics", action="store_false")
    ap.add_argument("--save", type=Path, nargs="?", const=DEFAULT_MODEL,
                    help=f"train one model on every clip (seed = first of --seeds) and save it (default {DEFAULT_MODEL})")
    args = ap.parse_args()

    rows = [r for d in args.dataset for r in load_rows(d)]
    for i, r in enumerate(rows):
        r["idx"] = i
    clips = load_clips(rows)
    bank = log_filterbank()

    if args.save:
        started = time.perf_counter()
        model, board, stats = train_fold(rows, clips, bank, args, args.seeds[0])
        save_model(args.save, model, board, stats, bank)
        print(f"Trained on all {len(rows)} clips ({len(board.positions)} string/fret positions) in "
              f"{time.perf_counter() - started:.0f} s; saved {args.save}")
        return

    if args.test:
        folds = {TEST_GROUP: split_by_groups(rows, {TEST_GROUP})}
    elif args.val_sessions:
        train, val = split_by_groups(rows, set(args.val_sessions))
        folds = {"+".join(args.val_sessions): ([r for r in train if r["group"] != TEST_GROUP], val)}
    else:
        dev = [r for r in rows if r["group"] != TEST_GROUP]
        folds = {g: split_by_groups(dev, {g}) for g in sorted({r["group"] for r in dev})}

    print(f"{len(rows)} clips; folds: {', '.join(folds)}; seeds {args.seeds}; {args.epochs} epochs; "
          f"augment {'on' if args.augment else 'off'}, harmonic features {'on' if args.harmonics else 'off'}")

    results = {fold: [] for fold in folds}
    confusion = torch.zeros(6, 6, dtype=torch.long)
    for seed in args.seeds:
        for fold, (train, val) in folds.items():
            started = time.perf_counter()
            model, board, stats = train_fold(train, clips, bank, args, seed)
            seen = set(board.positions)
            kept = [r for r in val if (r["string"], r["fret"]) in seen]
            result = score(kept, predict(model, board, stats, clips[[r["idx"] for r in kept]], bank))
            results[fold].append(result)
            confusion += result["confusion"]
            skipped = f", {len(val) - len(kept)} with unseen positions skipped" if len(kept) < len(val) else ""
            print(f"seed {seed}  {fold:<15}" + "".join(f"  {m} {result[m]:.1%}" for m in METRICS)
                  + f"   ({len(train)} train, {len(kept)} val{skipped}, {time.perf_counter() - started:.0f} s)")
    report(results, confusion, args.seeds)


if __name__ == "__main__":
    main()
