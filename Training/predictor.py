"""Pick the string for one plucked note with a model saved by `Training.py --save`."""
import numpy as np
import torch

from data import ONSET_S, SAMPLE_RATE, STORE_SAMPLES
from features import N_HARMONIC_FEATURES, log_filterbank, spectrogram, standardized_harmonics
from model import Fretboard, NoteNet

PRE_SAMPLES = int(ONSET_S * SAMPLE_RATE)
POST_SAMPLES = STORE_SAMPLES - PRE_SAMPLES


class StringPredictor:
    def __init__(self, path):
        saved = torch.load(path, weights_only=True)
        self.bank = log_filterbank()
        if saved["n_bands"] != self.bank.shape[0]:
            raise ValueError(f"{path} was trained with different spectrogram settings; retrain it with --save")
        self.board = Fretboard({tuple(p) for p in saved["positions"]})
        use_harmonics = saved["harmonic_mean"] is not None
        self.stats = (saved["harmonic_mean"], saved["harmonic_std"]) if use_harmonics else None
        self.model = NoteNet(self.bank.shape[0], len(self.board.pitches),
                             N_HARMONIC_FEATURES if use_harmonics else 0)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, clip, midi):
        """(string number, probability) for a clip that starts PRE_SAMPLES before the pluck,
        choosing only among strings that played this pitch in training. None if the model
        never saw this pitch."""
        pitch_id = self.board.pitch_ids.get(midi)
        if pitch_id is None:
            return None
        audio = torch.zeros(1, STORE_SAMPLES)
        clip = torch.from_numpy(np.asarray(clip, np.float32))[:STORE_SAMPLES]
        audio[0, :len(clip)] = clip
        audio = audio / audio.abs().max().clamp(min=1e-6)
        with torch.no_grad():
            embedding = self.model.embed(spectrogram(audio, self.bank))
            harmonics = standardized_harmonics(audio, torch.tensor([midi]), self.stats)
            logits = self.model.string_logits(embedding, harmonics)[0]
        probs = logits.masked_fill(~self.board.allowed[pitch_id], float("-inf")).softmax(dim=0)
        string_index = int(probs.argmax())
        return string_index + 1, float(probs[string_index])
