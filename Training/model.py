"""Pitch and string classifier, and decoding to a playable string/fret position."""
import torch
from torch import nn

from data import OPEN_MIDI, midi_of


class NoteNet(nn.Module):
    """Small CNN over the spectrogram with two outputs: pitch (one class per MIDI note) and
    string (6 classes). The CNN averages over time but keeps the frequency axis, because
    absolute pitch is what the pitch output needs. The string output also sees the harmonic
    features, which describe the tone of the note."""

    def __init__(self, n_bands, n_pitches, n_harmonic_features):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1)),
                                 nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))

        self.conv = nn.Sequential(block(1, 16), block(16, 32), block(32, 32))
        self.dropout = nn.Dropout(0.3)
        width = 32 * (n_bands // 8)
        self.pitch_head = nn.Linear(width, n_pitches)
        self.string_head = nn.Sequential(nn.Linear(width + n_harmonic_features, 64), nn.ReLU(),
                                         nn.Dropout(0.3), nn.Linear(64, 6))

    def embed(self, spec):
        return self.dropout(self.conv(spec).mean(dim=3).flatten(1))

    def string_logits(self, embedding, harmonics):
        return self.string_head(torch.cat([embedding, harmonics], dim=1))


class Fretboard:
    """The pitches and string/fret positions seen in training, and which strings may play
    each pitch. Decoding only ever returns one of these positions."""

    def __init__(self, positions):
        self.positions = sorted(positions)
        self.pitches = sorted({midi_of(s, f) for s, f in self.positions})
        self.pitch_ids = {m: i for i, m in enumerate(self.pitches)}
        self.allowed = torch.zeros(len(self.pitches), 6, dtype=torch.bool)
        for s, f in self.positions:
            self.allowed[self.pitch_ids[midi_of(s, f)], s - 1] = True

    def pitch_midi(self, pitch_ids):
        return torch.tensor(self.pitches)[pitch_ids]

    def decode(self, pitch_ids, string_logits):
        """(string, fret) per clip: the predicted pitch on the likeliest string that plays it."""
        masked = string_logits.masked_fill(~self.allowed[pitch_ids], float("-inf"))
        strings = masked.argmax(dim=1) + 1
        midi = self.pitch_midi(pitch_ids)
        return [(s, m - OPEN_MIDI[s]) for s, m in zip(strings.tolist(), midi.tolist())]
