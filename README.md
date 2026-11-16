# Guitar Analyzer

A real-time, string-aware guitar monitor. It listens to an acoustic guitar through a single pickup, works out which **string and fret** produced each note, and writes the result as tablature while you play.

Most pitch trackers only tell you *which note* sounded. On a guitar, the same pitch can come from up to three strings (A2 is the open A string or the 5th fret of the low E), so the monitor combines signal processing with small neural networks trained on single-note recordings to choose the string.

## Current status

The project is being deliberately narrowed. The current focus is **single notes in first position (frets 0–5)** on one acoustic guitar, and in particular **notes plucked while other strings are still ringing**, the biggest source of errors in live testing.

| Part | State |
|---|---|
| Pluck detection and pitch (DSP) | Works on isolated notes; misses some soft plucks over ringing strings |
| String/fret model (single notes) | 83.7% exact across held-out acoustic guitars; ~98% correct string when the pitch is right |
| Live tablature (ASCII and MusicXML export) | Working |
| Chord shapes (template fit and chord model) | Working on mixed test strums (~82% exact shapes on the own guitar); paused while single notes are improved |
| Notes over ringing strings | Being measured with recorded let-ring exercises (see [Evaluation](#evaluation)) |

## Hardware

- Acoustic guitar with a piezo / contact pickup.
- Behringer UM2 (or any USB audio interface); the instrument input is usually channel 1 (0-based).
- 48 kHz audio. Windows 11 is the tested platform; on Windows the UM2 opens most reliably through the WDM-KS "USB Audio CODEC" entry.

## Quick start

Double-click **`Guitar Analyzer.bat`**. On first run it creates a Python virtual environment (`.venv`) and installs `requirements.txt`, then opens the launcher. Python 3.10 or newer is required.

The launcher lets you pick the audio input once, then start:

- **Live monitor**: per-string display, spectrum, and live tablature from the guitar.
- **Simulator monitor**: the same display driven by a synthetic guitar (no hardware needed).
- **File monitor**: replay a WAV/FLAC recording.
- **Recorder**: prompted sessions that save labeled training clips.
- **Benchmark**: score the DSP analyzer against the simulator's ground truth.

Options in the launcher: **Trained models** (use `models/string_model.pt` when present) and **Mode**:

- **Picking**: every pluck is one note (recommended for now).
- **Chords**: every pluck is a strum.
- **Auto**: decides from how many pitches are heard.

The mode can also be changed in the monitor window while playing.

## Command line

```powershell
# Monitor
python run_hmi.py                                       # simulated guitar
python run_hmi.py --source live --list-devices
python run_hmi.py --source live --device 19 --channel 1 --model --mode picking
python run_hmi.py --source file --file recording.wav --model

# Labeled training clips (prompted)
python run_recorder.py --source live --device 19 --channel 1

# Let-ring exercises with known tab, for scoring notes over ringing strings
python record_exercise.py --device 19 --channel 1
```

Close the monitor before recording; the UM2 input usually can't be shared between programs.

## How it works

1. **Pluck detection**: a rise in high-frequency energy marks each attack (`guitar_hmi/analysis.py`).
2. **Pitch**: an iterative harmonic-salience search finds the pitches present just after the pluck; only pitches that got louder at the pluck count as new.
3. **String choice**: for single notes, the trained string model (`Training/model.py`, run through `Training/predictor.py`) reads a log-band spectrogram and harmonic features of the first 250 ms after the pluck and scores each string that can play the pitch.
4. **Fingering context**: a short Viterbi tracker (`guitar_hmi/fingering.py`) prefers hand positions that are physically comfortable given the previous notes, and can revise earlier notes.
5. **Chords**: either a non-negative fit of measured per-position spectra (`guitar_hmi/chord_fit.py`) or a per-string chord model trained on strums mixed from single notes (`Training/chords.py`), constrained to playable shapes.
6. **Tablature**: `guitar_hmi/tab.py` builds the tab live and exports ASCII or MusicXML to `tabs/`.

## Training

Training uses PyTorch on the CPU. Datasets and trained models are not stored in the repository (see `.gitignore`); rebuild them locally.

```powershell
python import_hf_dataset.py                 # Hugging Face clips -> dataset_hf/
python Training\Training.py                 # cross-validate, one guitar held out at a time
python Training\Training.py --save          # train the string model -> models/string_model.pt
python Training\templates.py                # chord templates -> models/chord_templates.npz
python Training\train_chords.py --save      # chord model -> models/chord_model.pt
```

- **Data**: own recordings (`dataset/`, from the recorder) plus the acoustic guitars from the Hugging Face set. Each source counts as one "guitar group" for cross-validation. `hf_deb` is a reserved test guitar that is never trained on.
- **Electric guitars**: the electric Hugging Face guitars are left out by default because the monitor listens to an acoustic. Add `--include-electric` to any training script to use them.
- **Fret range**: frets above `--max-fret` (default 5) are excluded.
- **Augmentation**: pitch warp, EQ, noise and mains hum.

## Evaluation

```powershell
python Training\eval_exercises.py                  # score recorded let-ring exercises
python Training\eval_exercises.py --synthetic 60   # random let-ring sequences mixed from your clips
python Training\eval_exercises.py --analyzer dsp   # DSP baseline without the string model
python Training\eval_chords.py                     # template fit vs. chord model
python sim_benchmark.py --events 200 --seed 1      # DSP analyzer on the simulator
```

`eval_exercises.py` replays each take in picking mode, lines the tabbed notes up with the expected notes, and reports how many notes were found and how many had the right pitch and the right string and fret. It breaks these down by what was ringing when each note was plucked: nothing, other strings, or other strings sharing overtones (unison, fifth, octave).

Baseline on 40 synthetic let-ring sequences (320 notes):

| Context | Found | Right pitch | Right string + fret |
|---|---|---|---|
| Nothing ringing | 100% | 98% | 98% |
| Over ringing strings | 89% | 84% | 83% |
| Over ringing, shared overtones | 82% | 71% | 67% |

Most losses come from missed plucks and wrong pitches, not from choosing the wrong string.

## Project layout

```
Guitar Analyzer.bat     one-click start (creates .venv on first run)
launcher.py             start-up window for every tool
run_hmi.py              real-time monitor
run_recorder.py         prompted recorder for labeled clips
record_exercise.py      let-ring exercise recorder (known tab)
import_hf_dataset.py    Hugging Face dataset import
sim_benchmark.py        DSP analyzer vs. simulator ground truth
guitar_hmi/             monitor: analysis, model analyzer, chord fit, fingering, tab, UI, audio sources
Training/               data loading, features, models, training and evaluation scripts
```

**String numbering:** `labels.csv` and the training code use string 1 = high E to 6 = low E. Inside `guitar_hmi`, string index 0 = low E.

## Roadmap

1. Record the let-ring exercises and score the current monitor on real audio.
2. Make pluck detection reliable for soft notes over ringing strings.
3. Explain each new pluck as "the ringing notes plus one new note" with the measured templates, to fix pitch errors over shared overtones.
4. If real recordings show string errors, train the string model with ringing-note tails mixed in and give it the spectrum from before the pluck.
5. Return to chords, then extend beyond first position.

## Data credits

The Hugging Face clips come from [collegefishiesd/guitar-fretboard-notes](https://huggingface.co/datasets/collegefishiesd/guitar-fretboard-notes) (CC BY-SA 4.0).
