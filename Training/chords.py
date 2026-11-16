"""Chords from single notes: a strum mixer, a per-string chord model, and decoding to a playable shape.

Real labeled strums are scarce, so training strums are mixed from single-note clips of one guitar:
one clip per string of a shape, each delayed in strum order and scaled, then summed. The model
predicts, for every string, one of N_CLASSES: not sounding, or fret 0..MAX_FRET.

Strings are indexed 0..5 from low E to high e here, as in guitar_hmi (labels.csv numbers them
the other way, 1 = high E).
"""
import csv
import itertools
import random
import sys

import numpy as np
import torch
from torch import nn

from augment import augment
from data import MAX_FRET, ROOT, SAMPLE_RATE, STORE_SAMPLES, load_audio
from features import FEATURE_S, log_filterbank, spectrogram

sys.path.insert(0, str(ROOT))
from guitar_hmi.chord_fit import playable  # noqa: E402
from guitar_hmi.guitar import VOICINGS  # noqa: E402

N_CLASSES = MAX_FRET + 2  # 0 = not sounding, 1 + fret
FRET_COLUMNS = ["fret_E", "fret_A", "fret_D", "fret_G", "fret_B", "fret_e"]
SHAPE_NAMES = {tuple((i, f) for i, f in enumerate(v) if f is not None): k for k, v in VOICINGS.items()}

STRUM_SPREAD_S = 0.03
P_VOICING = 0.5
P_DOWN = 0.7
GAIN_DB = (-6.0, 0.0)


def chord_rows(dataset_dir):
    """Clean labeled strums from labels.csv, each with frets {string index: fret}."""
    with open(dataset_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["kind"] == "chord" and r["quality"] == "clean" and r["qc_pass"] == "1"]
    for r in rows:
        r["path"] = dataset_dir / r["file"]
        r["frets"] = {i: int(r[c]) for i, c in enumerate(FRET_COLUMNS) if int(r[c]) >= 0}
        r["group"] = r["session"] if r["session"].startswith("hf_") else "piezo"
    return rows


def load_chord_clips(rows):
    return torch.stack([load_audio(r) for r in rows]) if rows else torch.zeros(0, STORE_SAMPLES)


def targets_of(frets):
    return [1 + frets[i] if i in frets else 0 for i in range(6)]


def frets_of(target):
    return {i: int(c) - 1 for i, c in enumerate(target) if c > 0}


class StrumMixer:
    """Synthetic strums from one guitar group's single-note clips (rows from data.load_rows,
    each with an "idx" into clips)."""

    def __init__(self, rows, clips, groups):
        self.clips = clips
        self.by_group = {}
        for r in rows:
            if r["group"] in groups:
                key = (6 - r["string"], r["fret"])
                self.by_group.setdefault(r["group"], {}).setdefault(key, []).append(r["idx"])
        self.groups = sorted(self.by_group)
        self.voicings = {g: [dict(s) for s in SHAPE_NAMES if all(p in have for p in s)]
                         for g, have in self.by_group.items()}

    def _random_shape(self, have, rng):
        low = rng.randrange(5)
        high = rng.randrange(low + 1, 6)
        base = rng.randrange(MAX_FRET - 1)
        frets = {}
        for s in range(low, high + 1):
            options = [f for f in [0] + list(range(base + 1, base + 4)) if (s, f) in have]
            if not options:
                break
            frets[s] = rng.choice(options)
        return frets

    def shape(self, group, rng):
        have = self.by_group[group]
        if self.voicings[group] and rng.random() < P_VOICING:
            return dict(rng.choice(self.voicings[group]))
        for _ in range(20):
            frets = self._random_shape(have, rng)
            if playable(frets):
                return frets
        return dict(rng.choice(self.voicings[group])) if self.voicings[group] else {}

    def strum(self, group, frets, rng):
        """(audio (STORE_SAMPLES,), target (6,)) for one strum of frets."""
        order = sorted(frets) if rng.random() < P_DOWN else sorted(frets, reverse=True)
        spread = rng.uniform(0, STRUM_SPREAD_S)
        out = torch.zeros(STORE_SAMPLES)
        for k, s in enumerate(order):
            clip = self.clips[rng.choice(self.by_group[group][(s, frets[s])])]
            delay = int(spread * k / max(len(order) - 1, 1) * SAMPLE_RATE)
            out[delay:] += clip[:STORE_SAMPLES - delay] * 10 ** (rng.uniform(*GAIN_DB) / 20)
        return out / out.abs().max().clamp(min=1e-6), torch.tensor(targets_of(frets))

    def batch(self, n, rng, groups=None):
        audio, target = [], []
        for _ in range(n):
            group = rng.choice(groups or self.groups)
            a, t = self.strum(group, self.shape(group, rng), rng)
            audio.append(a)
            target.append(t)
        return torch.stack(audio), torch.stack(target)


class ChordNet(nn.Module):
    """The NoteNet convolutional trunk with one N_CLASSES output per string."""

    def __init__(self, n_bands):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1)),
                                 nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))

        self.conv = nn.Sequential(block(1, 16), block(16, 32), block(32, 32))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(32 * (n_bands // 8), 6 * N_CLASSES)

    def forward(self, spec):
        return self.head(self.dropout(self.conv(spec).mean(dim=3).flatten(1))).view(-1, 6, N_CLASSES)


def _all_shapes():
    shapes = [c for c in itertools.product(range(N_CLASSES), repeat=6) if playable(frets_of(c))]
    return torch.tensor(shapes)


SHAPES = _all_shapes()


def decode(log_probs):
    """Most probable playable shape for each strum's (6, N_CLASSES) log-probabilities:
    (targets (batch, 6), joint probability (batch,))."""
    out, joint = [], []
    for lp in log_probs:
        score = lp[torch.arange(6), SHAPES].sum(dim=1)
        k = int(score.argmax())
        out.append(SHAPES[k])
        joint.append(float(score[k].exp()))
    return torch.stack(out), torch.tensor(joint)


def train_chord_model(mixer, groups, epochs, steps, batch_size, lr, seed, use_augment=True):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    bank = log_filterbank()
    model = ChordNet(bank.shape[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    model.train()
    for _ in range(epochs):
        for _ in range(steps):
            audio, target = mixer.batch(batch_size, rng, groups)
            if use_augment:
                audio = augment(audio)
            logits = model(spectrogram(audio, bank))
            loss = loss_fn(logits.reshape(-1, N_CLASSES), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
    model.eval()
    return model


def save_chord_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_bands": log_filterbank().shape[0],
                "feature_s": FEATURE_S}, path)


def score_shapes(truth, predicted):
    """Exact-shape, per-string and chord-name accuracy for lists of {string index: fret};
    chord name counts only strums whose true shape is a library voicing."""
    n = len(truth)
    named = [(t, p) for t, p in zip(truth, predicted) if tuple(sorted(t.items())) in SHAPE_NAMES]
    return {
        "exact": sum(t == p for t, p in zip(truth, predicted)) / n,
        "per_string": sum(t.get(s) == p.get(s) for t, p in zip(truth, predicted) for s in range(6)) / (6 * n),
        "name": (sum(SHAPE_NAMES[tuple(sorted(t.items()))] == SHAPE_NAMES.get(tuple(sorted(p.items())))
                     for t, p in named) / len(named)) if named else float("nan"),
    }


def predict_shapes(model, audio, batch=64):
    bank = log_filterbank()
    targets, joints, probs = [], [], []
    with torch.no_grad():
        for a in audio.split(batch):
            lp = model(spectrogram(a, bank)).log_softmax(dim=2)
            t, j = decode(lp)
            targets.append(t)
            joints.append(j)
            probs.append(lp.exp())
    return torch.cat(targets), torch.cat(joints), torch.cat(probs)


class ChordPredictor:
    """A saved ChordNet for the monitor: clip in (ONSET_S before the strum to POST_S after),
    playable shape out."""

    def __init__(self, path):
        saved = torch.load(path, weights_only=True)
        if saved["n_bands"] != log_filterbank().shape[0] or saved.get("feature_s") != FEATURE_S:
            raise ValueError(f"{path} was trained with different feature settings; retrain it with "
                             "Training\\train_chords.py --save")
        self.model = ChordNet(saved["n_bands"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, clip):
        """({string index: fret}, {string index: probability of that class}, joint probability)."""
        audio = torch.zeros(1, STORE_SAMPLES)
        clip = torch.from_numpy(np.asarray(clip, np.float32))[:STORE_SAMPLES]
        audio[0, :len(clip)] = clip
        audio = audio / audio.abs().max().clamp(min=1e-6)
        target, joint, probs = predict_shapes(self.model, audio)
        frets = frets_of(target[0].tolist())
        sure = {i: float(probs[0, i, target[0, i]]) for i in range(6)}
        return frets, sure, float(joint[0])
