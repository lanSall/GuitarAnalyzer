"""String probabilities for one plucked note, from a model saved by `Training.py --save`."""
import numpy as np
import torch

from data import ONSET_S, SAMPLE_RATE, STORE_SAMPLES
from features import FEATURE_S, N_HARMONIC_FEATURES, log_filterbank, spectrogram, standardized_harmonics
from model import Fretboard, NoteNet

PRE_SAMPLES = int(ONSET_S * SAMPLE_RATE)
POST_SAMPLES = STORE_SAMPLES - PRE_SAMPLES


class StringPredictor:
    def __init__(self, path):
        saved = torch.load(path, weights_only=True)
        self.bank = log_filterbank()
        if saved["n_bands"] != self.bank.shape[0] or saved.get("feature_s") != FEATURE_S:
            raise ValueError(f"{path} was trained with different feature settings; retrain it with "
                             "Training.py --save")
        self.board = Fretboard({tuple(p) for p in saved["positions"]})
        self.max_fret = max(fret for _, fret in self.board.positions)
        use_harmonics = saved["harmonic_mean"] is not None
        self.stats = (saved["harmonic_mean"], saved["harmonic_std"]) if use_harmonics else None
        self.model = NoteNet(self.bank.shape[0], len(self.board.pitches),
                             N_HARMONIC_FEATURES if use_harmonics else 0)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def string_probs(self, clip, midi):
        """{string number: probability} for a clip that starts PRE_SAMPLES before the pluck and
        runs POST_SAMPLES after it, over the strings that played this pitch in training. None if
        the model never saw this pitch."""
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
        allowed = self.board.allowed[pitch_id]
        probs = logits.masked_fill(~allowed, float("-inf")).softmax(dim=0)
        return {i + 1: float(probs[i]) for i in range(6) if allowed[i]}
