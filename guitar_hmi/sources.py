"""Audio sources that feed a shared ring buffer: simulator, live input, WAV file."""
import os
import threading
import time
from collections import deque

import numpy as np

from .guitar import OPEN_MIDI, VOICINGS, midi_to_hz, note_name, string_number, tab_string


class RingBuffer:
    """Mono sample buffer addressed by absolute sample index."""

    def __init__(self, sr, seconds=10.0):
        self.sr = sr
        self.size = int(sr * seconds)
        self._buf = np.zeros(self.size, np.float32)
        self.total = 0
        self._lock = threading.Lock()

    def write(self, x):
        x = np.asarray(x, np.float32).ravel()
        for i in range(0, len(x), self.size):
            chunk = x[i:i + self.size]
            n = len(chunk)
            with self._lock:
                start = self.total % self.size
                end = start + n
                if end <= self.size:
                    self._buf[start:end] = chunk
                else:
                    k = self.size - start
                    self._buf[start:] = chunk[:k]
                    self._buf[:n - k] = chunk[k:]
                self.total += n

    def get_range(self, a, b):
        """Samples [a, b). Indices not yet written or already overwritten read as zero."""
        out = np.zeros(max(b - a, 0), np.float32)
        with self._lock:
            lo = max(a, self.total - self.size, 0)
            hi = min(b, self.total)
            if hi > lo:
                out[lo - a:hi - a] = self._buf[np.arange(lo, hi) % self.size]
        return out

    def latest(self, n):
        total = self.total
        return self.get_range(total - n, total)


# Per-string inharmonicity; plain strings (G, B, high E) are stiffer for their pitch.
_INHARMONICITY = [1.0e-4, 0.9e-4, 0.8e-4, 2.4e-4, 1.8e-4, 1.3e-4]


def synth_note(midi, string_index, sr, rng, quality="clean", velocity=1.0):
    """Additive model of a plucked string as heard through a piezo."""
    f0 = float(midi_to_hz(midi))
    dur = 0.7 if quality == "muted" else 3.0
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, np.float32)
    beta = _INHARMONICITY[string_index]
    pluck_pos = rng.uniform(0.12, 0.25)
    tau1 = rng.uniform(1.2, 2.0) * (200.0 / f0) ** 0.4
    for h in range(1, 31):
        fh = h * f0 * np.sqrt(1.0 + beta * h * h)
        if fh > 0.45 * sr or fh > 9000:
            break
        amp = abs(np.sin(np.pi * h * pluck_pos)) / h ** 1.1
        tau = tau1 / (1.0 + 0.25 * (h - 1))
        if quality == "muted":
            tau = 0.06 / (1.0 + 0.15 * (h - 1))
        m = min(n, int(7 * tau * sr) + 1)
        out[:m] += amp * np.exp(-t[:m] / tau) * np.sin(2 * np.pi * fh * t[:m] + rng.uniform(0, 2 * np.pi))
    out *= 1.0 - np.exp(-t / 0.0015)
    m = int(0.03 * sr)
    out[:m] += 0.05 * rng.standard_normal(m).astype(np.float32) * np.exp(-t[:m] / 0.004)
    if quality == "buzz":
        rattle = np.diff(rng.standard_normal(n + 1)).astype(np.float32)
        contact = (np.sin(2 * np.pi * f0 * t) > 0.6).astype(np.float32)
        out += 0.12 * rattle * contact * np.exp(-t / 0.35)
    out /= np.max(np.abs(out)) + 1e-9
    return out * (0.25 * velocity)


class GuitarSynth:
    """Plays random single notes and chord strums (some with mistakes) and records ground truth."""

    def __init__(self, sr, seed=None, mistake_rate=0.25):
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.mistake_rate = mistake_rate
        self._voices = []
        self._prepared = None
        self.truth = []

    def prepare(self):
        """Choose and synthesize the next event ahead of time (synthesis is the slow part)."""
        rng, sr = self.rng, self.sr
        if rng.random() < 0.45:
            s = int(rng.integers(6))
            frets = {s: int(rng.integers(0, 13))}
            order = [s]
            chord = None
        else:
            chord = str(rng.choice(list(VOICINGS)))
            frets = {i: f for i, f in enumerate(VOICINGS[chord]) if f is not None}
            order = sorted(frets, reverse=rng.random() < 0.3)
        mistakes = {}
        if rng.random() < self.mistake_rate:
            mistakes[int(rng.choice(list(frets)))] = str(rng.choice(["muted", "buzz"]))

        step = rng.uniform(0.008, 0.018) * sr
        velocity = rng.uniform(0.5, 1.0)
        voices = []
        for k, s in enumerate(order):
            audio = synth_note(OPEN_MIDI[s] + frets[s], s, sr, rng, mistakes.get(s, "clean"), velocity)
            voices.append({"audio": audio, "t": -int(k * step), "gain": 1.0, "release": False})

        if chord:
            label = f"strum {chord}  {tab_string(frets)}"
        else:
            s, f = next(iter(frets.items()))
            label = f"string {string_number(s)} fret {f}  ({note_name(OPEN_MIDI[s] + f)})"
        if mistakes:
            label += "  [" + ", ".join(f"string {string_number(s)} {q}" for s, q in mistakes.items()) + "]"
        self._prepared = (voices, {"frets": frets, "mistakes": mistakes, "label": label})

    def trigger(self, now):
        if self._prepared is None:
            self.prepare()
        voices, truth = self._prepared
        self._prepared = None
        for v in self._voices:
            v["release"] = True
        self._voices += voices
        self.truth.append({"time": now / self.sr, **truth})

    def render(self, n):
        out = self.rng.normal(0.0, 1e-4, n).astype(np.float32)
        fade_step = 1.0 / (0.03 * self.sr)
        alive = []
        for v in self._voices:
            a, t = v["audio"], v["t"]
            lo, hi = max(t, 0), min(t + n, len(a))
            if hi > lo:
                seg = a[lo:hi]
                if v["release"]:
                    gain = v["gain"] - fade_step * np.arange(hi - lo)
                    seg = seg * np.clip(gain, 0.0, 1.0)
                    v["gain"] = gain[-1] - fade_step
                out[lo - t:hi - t] += seg
            v["t"] = t + n
            if v["t"] < len(a) and v["gain"] > 0:
                alive.append(v)
        self._voices = alive
        return out


class AudioSource:
    label = ""

    def __init__(self, sr):
        self.sr = sr
        self.ring = RingBuffer(sr)
        self.truth = deque()

    def start(self):
        pass

    def stop(self):
        pass


class _PacedSource(AudioSource):
    """Writes generated audio into the ring buffer at real-time speed from a worker thread."""

    block = 256

    def __init__(self, sr):
        super().__init__(sr)
        self._running = False
        self._thread = None

    def _next_block(self, written):
        raise NotImplementedError

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        t0 = time.perf_counter()
        written = 0
        while self._running:
            target = int((time.perf_counter() - t0) * self.sr)
            while self._running and written + self.block <= target:
                self.ring.write(self._next_block(written))
                written += self.block
            time.sleep(0.003)


class SimulatedSource(_PacedSource):
    label = "Simulated guitar"

    def __init__(self, sr=48000, seed=None, mistake_rate=0.25):
        super().__init__(sr)
        self.synth = GuitarSynth(sr, seed, mistake_rate)
        self._next_event = int(1.0 * sr)
        self._prepare_at = 0
        self._reported = 0

    def _next_block(self, written):
        if written >= self._next_event:
            self.synth.trigger(written)
            self._next_event = written + int(self.synth.rng.uniform(1.4, 2.6) * self.sr)
            self._prepare_at = written + int(0.7 * self.sr)
        elif written >= self._prepare_at:
            self.synth.prepare()
            self._prepare_at = 10 ** 15
        block = self.synth.render(self.block)
        while self._reported < len(self.synth.truth):
            self.truth.append(self.synth.truth[self._reported])
            self._reported += 1
        return block


class FileSource(_PacedSource):
    def __init__(self, path, channel=0, loop=True):
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        super().__init__(sr)
        self.data = data[:, min(channel, data.shape[1] - 1)]
        self.loop = loop
        self.label = f"File: {os.path.basename(path)}"

    def _next_block(self, written):
        pos = written % len(self.data) if self.loop else written
        block = self.data[pos:pos + self.block]
        if len(block) < self.block:
            block = np.pad(block, (0, self.block - len(block)))
        return block


class LiveSource(AudioSource):
    def __init__(self, sr=48000, device=None, channel=0, blocksize=128):
        super().__init__(sr)
        self.device = device
        self.channel = channel
        self.blocksize = blocksize
        self._stream = None

    def start(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            self.ring.write(indata[:, self.channel])

        self._stream = sd.InputStream(
            samplerate=self.sr,
            device=self.device,
            channels=self.channel + 1,
            dtype="float32",
            blocksize=self.blocksize,
            latency="low",
            callback=callback,
        )
        self._stream.start()
        info = sd.query_devices(self._stream.device)
        api = sd.query_hostapis(info["hostapi"])["name"]
        self.label = f"Live: {info['name']} ({api}), ch {self.channel}, {self._stream.latency * 1000:.1f} ms"

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
