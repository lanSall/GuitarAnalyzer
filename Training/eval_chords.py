"""Compare the template fitter and the chord model on the same strums.

    python Training\\eval_chords.py
    python Training\\eval_chords.py --folds piezo      only the own-guitar test

Folds:
- each Hugging Face guitar held out: neither method has heard that guitar (fitter templates and
  model training come from the other guitars); test strums are mixed from its clips.
- piezo: the realistic case for the monitor. Piezo takes are split in half per position; the
  fitter's piezo templates and the model's piezo strums come from one half, test strums from
  the other.
- real strums: labeled chords in labels.csv, with the saved templates and chord model.
"""
import argparse
import collections
import sys
import time

import torch

from chords import StrumMixer, frets_of, predict_shapes, score_shapes, train_chord_model
from data import ONSET_S, ROOT, SAMPLE_RATE
from templates import DEFAULT_PATH as TEMPLATES, build_templates
from train_chords import DEFAULT_MODEL, add_training_args, load_notes, real_strums, report, test_strums

sys.path.insert(0, str(ROOT))
from guitar_hmi.chord_fit import ChordFitter  # noqa: E402

HF_FOLDS = ["hf_ele", "hf_ele_natural", "hf_eqm", "hf_eqm2"]  # the electric ones only with --include-electric
PRE_SAMPLES = int(ONSET_S * SAMPLE_RATE)


def fit_all(fitter, audio):
    out = []
    for a in audio.numpy():
        fit = fitter.fit(a, PRE_SAMPLES)
        out.append(fit.frets if fit else {})
    return out


def fitter_for(rows, clips, group):
    templates = build_templates(rows, clips[[r["idx"] for r in rows]],
                                pooled_from={r["group"] for r in rows if r["group"] != "piezo"})
    fitter = ChordFitter(TEMPLATES, group=group)
    fitter.set_templates(templates.get(group, templates["pooled"])[0].T)
    return fitter


def run_fold(name, train_rows, test_rows, test_group, fitter_group, clips, args):
    train_mixer = StrumMixer(train_rows, clips, {r["group"] for r in train_rows})
    test_mixer = StrumMixer(test_rows, clips, {test_group})
    audio, truth = test_strums(test_mixer, test_group, args.test_strums, args.seed + 1)

    started = time.perf_counter()
    fitter = fitter_for(train_rows, clips, fitter_group)
    report(f"{name}: template fit", score_shapes(truth, fit_all(fitter, audio)))
    model = train_chord_model(train_mixer, train_mixer.groups, args.epochs, args.steps, args.batch_size,
                              args.lr, args.seed)
    predicted = [frets_of(t.tolist()) for t in predict_shapes(model, audio)[0]]
    report(f"{name}: chord model", score_shapes(truth, predicted))
    print(f"    ({len(truth)} test strums, {time.perf_counter() - started:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description="Compare the template fitter and the chord model")
    add_training_args(ap)
    ap.add_argument("--folds", nargs="+", help="default: every Hugging Face development guitar loaded, and piezo")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows, clips = load_notes(args.dataset, args.include_electric)
    groups = {r["group"] for r in rows}
    args.folds = args.folds or [g for g in HF_FOLDS if g in groups] + ["piezo"]
    print(f"{len(rows)} single-note clips; {args.epochs} epochs x {args.steps} batches of {args.batch_size} strums")
    for fold in args.folds:
        if fold == "piezo":
            seen = collections.Counter()
            half = {}
            for r in rows:
                if r["group"] == "piezo":
                    key = (r["string"], r["fret"])
                    half[r["idx"]] = seen[key] % 2
                    seen[key] += 1
            train_rows = [r for r in rows if half.get(r["idx"], 0) == 0]
            test_rows = [r for r in rows if half.get(r["idx"]) == 1]
            run_fold("piezo, other takes", train_rows, test_rows, "piezo", "piezo", clips, args)
        else:
            train_rows = [r for r in rows if r["group"] != fold]
            test_rows = [r for r in rows if r["group"] == fold]
            run_fold(f"{fold} held out", train_rows, test_rows, fold, "pooled", clips, args)

    real, real_clips = real_strums(args.dataset)
    if not real:
        print("No real labeled strums yet (record some in the recorder's Chords mode).")
        return
    truth = [r["frets"] for r in real]
    report(f"real strums ({len(real)}): template fit",
           score_shapes(truth, fit_all(ChordFitter(TEMPLATES, group="piezo"), real_clips)))
    if DEFAULT_MODEL.exists():
        from chords import ChordPredictor

        model = ChordPredictor(DEFAULT_MODEL).model
        predicted = [frets_of(t.tolist()) for t in predict_shapes(model, real_clips)[0]]
        report(f"real strums ({len(real)}): chord model", score_shapes(truth, predicted))


if __name__ == "__main__":
    main()
