"""Labeled clips: reading labels.csv, grouping by guitar, splitting, and loading audio."""
import csv
from pathlib import Path

import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 48000
ONSET_S = 0.1
POST_S = 0.25
STORE_SAMPLES = int((ONSET_S + POST_S) * SAMPLE_RATE)
OPEN_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
TEST_GROUP = "hf_deb"
MAX_FRET = 5
# Hugging Face electric guitars, left out by default because the monitor listens to an acoustic.
ELECTRIC_GROUPS = {"hf_ele", "hf_ele_natural"}


def midi_of(string, fret):
    return OPEN_MIDI[string] + fret


def load_rows(dataset_dir, max_fret=MAX_FRET, electric=False):
    """Clean single-note clips at or below max_fret that passed the recorder's quality check,
    without the ELECTRIC_GROUPS unless electric is true."""
    with open(dataset_dir / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["kind"] == "note" and r["quality"] == "clean" and r["qc_pass"] == "1"]
    for r in rows:
        r["path"] = dataset_dir / r["file"]
        r["string"], r["fret"] = int(r["string"]), int(r["fret"])
        r["group"] = r["session"] if r["session"].startswith("hf_") else "piezo"
    return [r for r in rows if r["fret"] <= max_fret and (electric or r["group"] not in ELECTRIC_GROUPS)]


def split_by_groups(rows, held_out):
    """Rows whose guitar group or session is in held_out go to validation, the rest to training."""
    train = [r for r in rows if r["group"] not in held_out and r["session"] not in held_out]
    val = [r for r in rows if r["group"] in held_out or r["session"] in held_out]
    return train, val


def load_audio(row, n_samples=STORE_SAMPLES):
    """Mono clip scaled to a peak of 1 and shifted so the pluck lands at ONSET_S."""
    samples, sr = sf.read(row["path"], dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{row['file']}: expected {SAMPLE_RATE} Hz, got {sr} Hz")
    audio = torch.from_numpy(samples.mean(axis=1))
    audio = audio / audio.abs().max().clamp(min=1e-6)
    shift = round((ONSET_S - float(row["onset_s"])) * SAMPLE_RATE)
    audio = nn.functional.pad(audio, (max(shift, 0), 0))[max(-shift, 0):][:n_samples]
    return nn.functional.pad(audio, (0, n_samples - len(audio)))


def load_clips(rows, n_samples=STORE_SAMPLES):
    """(clips, samples) tensor holding every row's audio from ONSET_S before the pluck to POST_S
    after it, loaded once. Live use reads the same span from the ring buffer."""
    return torch.stack([load_audio(r, n_samples) for r in rows]) if rows else torch.zeros(0, n_samples)
