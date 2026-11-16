"""Per-position tone templates for the monitor's chord fitter, from the single-note clips.

    python Training\\templates.py        writes models/chord_templates.npz

A template is the average spectrum (guitar_hmi.chord_fit.BandSpectrum, each clip scaled to unit
length) of one guitar group's clips at one string and fret. A position the group never
recorded borrows the nearest recorded fret on the same string, shifted by the fret difference;
a string the group has no clips for at all uses the pooled templates of the other guitars.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from data import ONSET_S, ROOT, SAMPLE_RATE, load_clips, load_rows

sys.path.insert(0, str(ROOT))
from guitar_hmi.chord_fit import POSITIONS, BandSpectrum, shift_frets  # noqa: E402

DEFAULT_PATH = ROOT / "models" / "chord_templates.npz"
POOLED = "pooled"


def clip_spectra(clips):
    spectrum = BandSpectrum(SAMPLE_RATE)
    onset = int(ONSET_S * SAMPLE_RATE)
    out = np.stack([spectrum(c, onset) for c in clips.numpy()])
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _group_templates(spectra, positions):
    """(templates (len(POSITIONS), bands), clip counts) from (spectrum, (string index, fret)) pairs."""
    by = {}
    for spec, p in zip(spectra, positions):
        by.setdefault(p, []).append(spec)
    templates = np.zeros((len(POSITIONS), spectra.shape[1]), np.float32)
    counts = np.zeros(len(POSITIONS), np.int64)
    for k, (s, f) in enumerate(POSITIONS):
        if (s, f) in by:
            templates[k], counts[k] = np.mean(by[(s, f)], axis=0), len(by[(s, f)])
            continue
        recorded = [p for p in by if p[0] == s]
        if recorded:
            near = min(recorded, key=lambda p: abs(p[1] - f))
            templates[k] = shift_frets(np.mean(by[near], axis=0), f - near[1])
    return templates, counts


def build_templates(rows, clips, pooled_from=None):
    """{group: (templates (len(POSITIONS), bands), clip counts)}, plus POOLED over the groups in
    pooled_from (default: every group but piezo)."""
    spectra = clip_spectra(clips)
    positions = [(6 - r["string"], r["fret"]) for r in rows]
    group = np.array([r["group"] for r in rows])
    pooled_from = set(pooled_from or {g for g in group if g != "piezo"})
    use = np.isin(group, list(pooled_from))
    out = {POOLED: _group_templates(spectra[use], [p for p, u in zip(positions, use) if u])}
    for g in sorted(set(group)):
        use = group == g
        templates, counts = _group_templates(spectra[use], [p for p, u in zip(positions, use) if u])
        empty = ~templates.any(axis=1)
        templates[empty] = out[POOLED][0][empty]
        out[g] = (templates, counts)
    return out


def save_templates(path, templates):
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted(templates)
    np.savez(path, groups=np.array(groups), spectra=np.stack([templates[g][0] for g in groups]),
             counts=np.stack([templates[g][1] for g in groups]))


def main():
    ap = argparse.ArgumentParser(description="Build per-position tone templates for the chord fitter")
    ap.add_argument("--dataset", type=Path, nargs="+", default=[ROOT / "dataset", ROOT / "dataset_hf"])
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--include-electric", action="store_true", help="also pool the electric HF guitars")
    args = ap.parse_args()

    rows = [r for d in args.dataset for r in load_rows(d, electric=args.include_electric)]
    templates = build_templates(rows, load_clips(rows))
    save_templates(args.out, templates)
    print(f"Templates from {len(rows)} clips saved to {args.out}")
    for g, (_, counts) in templates.items():
        print(f"  {g:<15} {int((counts > 0).sum()):>2} of {len(POSITIONS)} positions recorded")


if __name__ == "__main__":
    main()
