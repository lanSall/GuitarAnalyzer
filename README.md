# Guitar Analyzer

A real-time, string-aware guitar monitor. It listens to an acoustic guitar through a single pickup, works out which **string and fret** produced each note, and writes the result as tablature while you play.

Most pitch trackers only tell you *which note* sounded. On a guitar, the same pitch can come from up to three strings (A2 is the open A string or the 5th fret of the low E), so the monitor combines signal processing with small neural networks trained on single-note recordings to choose the string.

## Current status

The project is being deliberately narrowed. The current focus is **single notes in first position (frets 0–5)** on one acoustic guitar, and in particular **notes plucked while other strings are still ringing**, the biggest source of errors in live testing.

| Part | State |
|---|---|
| Pluck detection (spectral flux) | Finds 99% of plucks in real let-ring exercises, including soft ones over ringing strings |
| Pitch (DSP, ringing subtracted) | 88% right on real let-ring exercises; the misses are mostly octave and two-octave errors |
| String/fret model (single notes) | Trained on single notes plus notes cut from let-ring exercises; on held-out exercise takes every note with the right pitch got the right string |
| Live tablature and sheet music (ASCII tab, MusicXML, PDF with notation over tab) | Working; rhythm is rounded to 8th notes and the key is guessed from the notes played |
| Chord shapes (template fit and chord model) | Working on mixed test strums (~82% exact shapes on the own guitar); paused while single notes are improved |
| Buzz / muted-string detection | Removed; it didn't add anything useful |

## Hardware

- Acoustic guitar with a piezo / contact pickup.
- Behringer UM2 (or any USB audio interface); the instrument (Hi-Z) input is channel 1 (0-based), shown as channel 2 in the launcher.
- 48 kHz audio. Windows 11 is the tested platform; on Windows the UM2 opens most reliably through the WDM-KS "Microphone (USB Audio CODEC)" entry. Device numbers change when hardware changes, so the launcher remembers the input by name, and `--device` accepts part of a name.
- Set the amp/interface gain so light plucks register; the pluck detector can't hear what never reaches it.

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
python run_hmi.py --source live --device "USB Audio CODEC" --channel 1 --model --mode picking
python run_hmi.py --source file --file recording.wav --model

# Labeled training clips (prompted)
python run_recorder.py --source live --device "USB Audio CODEC" --channel 1

# Let-ring exercises with known tab, for scoring and training on notes over ringing strings
python record_exercise.py --device "USB Audio CODEC" --channel 1
```

**Recording tips**

- In the clip recorder, keep **Shuffle** on: it records one take of every position per round, so each position is heard across the whole session instead of in one burst. To add takes to positions that already have plenty, uncheck "Only record what's still missing".
- For the let-ring exercises, play about **one note per second** and let every note ring into the next. The recorder warns when notes were more than 2 s apart, since the earlier strings will have faded and the take no longer tests ringing.

Close the monitor before recording; the UM2 input usually can't be shared between programs.

## How it works

1. **Pluck detection**: spectral flux over 24 bands per octave (60 Hz–8 kHz) against an adaptive threshold marks each attack, so a new note stands out even while other strings ring; the attack is then pinned to the first sharp high-frequency rise (`guitar_hmi/analysis.py`, `FluxOnsetDetector`).
2. **Pitch**: the spectrum from just before the pluck is subtracted from the one just after, so strings that were already ringing drop out; an iterative harmonic-salience search then finds the new pitches, and only pitches that got louder at the pluck count.
3. **String choice**: for single notes, the trained string model (`Training/model.py`, run through `Training/predictor.py`) reads a log-band spectrogram and harmonic features of the first 250 ms after the pluck and scores each string that can play the pitch.
4. **Fingering context**: a short Viterbi tracker (`guitar_hmi/fingering.py`) prefers hand positions that are physically comfortable given the previous notes, and can revise earlier notes.
5. **Chords**: either a non-negative fit of measured per-position spectra (`guitar_hmi/chord_fit.py`) or a per-string chord model trained on strums mixed from single notes (`Training/chords.py`), constrained to playable shapes.
6. **Tablature and sheet music**: `guitar_hmi/tab.py` builds the tab live. **Export tab** in the monitor saves ASCII tab, MusicXML (standard notation over a tab staff, opens in MuseScore and other notation programs) and a printable PDF engraved by [Verovio](https://www.verovio.org) (`guitar_hmi/sheet.py`) to `tabs/`. Any MusicXML file can be turned into a PDF with `python -m guitar_hmi.sheet file.musicxml`.

## Training

Training uses PyTorch on the CPU. Datasets and trained models are not stored in the repository (see `.gitignore`); rebuild them locally.

```powershell
python import_hf_dataset.py                 # Hugging Face clips -> dataset_hf/
python Training\Training.py                 # cross-validate, one guitar held out at a time
python Training\exercise_clips.py           # let-ring exercise takes -> labeled clips in dataset_ringing/
python Training\Training.py --save          # train the string model -> models/string_model.pt
python Training\templates.py                # chord templates -> models/chord_templates.npz
python Training\train_chords.py --save      # chord model -> models/chord_model.pt
```

- **Data**: own recordings (`dataset/`, from the recorder), notes cut from the let-ring exercises (`dataset_ringing/`, read automatically when present), and the acoustic guitars from the Hugging Face set. Each source counts as one "guitar group" for cross-validation. `hf_deb` is a reserved test guitar that is never trained on.
- **Why the exercise clips matter**: single-note clips never contain other strings ringing. Trained on those alone, the model put open A on the low E's 5th fret whenever the low E was still sounding. Notes cut from the exercises fixed that. `exercise_clips.py` only keeps notes the DSP heard at the right pitch, so a misheard note never becomes a mislabeled clip.
- **Keep a test set**: never score a model on exercise takes it trained on. `exercise_clips.py --takes 1 2` with `eval_exercises.py --takes 3` splits by take; better still, score a new day's takes before adding them to training.
- **Electric guitars**: the electric Hugging Face guitars are left out by default because the monitor listens to an acoustic. Add `--include-electric` to any training script to use them.
- **Fret range**: frets above `--max-fret` (default 5) are excluded.
- **Augmentation**: pitch warp, EQ, noise and mains hum.

## Evaluation

```powershell
python Training\eval_exercises.py                  # score recorded let-ring exercises
python Training\eval_exercises.py --takes 3        # only take 3 (the take left out of training)
python Training\eval_exercises.py --synthetic 60   # random let-ring sequences mixed from your clips
python Training\eval_exercises.py --analyzer dsp   # DSP baseline without the string model
python Training\eval_chords.py                     # template fit vs. chord model
python sim_benchmark.py --events 200 --seed 1      # DSP analyzer on the simulator
```

`eval_exercises.py` replays each take in picking mode, lines the tabbed notes up with the expected notes, and reports how many notes were found and how many had the right pitch and the right string and fret. It breaks these down by what was ringing when each note was plucked: nothing, other strings, or other strings sharing overtones (unison, fifth, octave).

Real let-ring exercises (12 takes, 111 notes, most plucked over one to three ringing strings):

| Version | Found | Right pitch | Right string + fret |
|---|---|---|---|
| Old pluck detector, no ringing subtraction, earlier model | 80% | 60% | 41% |
| Flux pluck detector + ringing subtraction | 99% | 88% | 77% |

Training on exercise clips, scored only on held-out takes:

| Test | Before | After |
|---|---|---|
| Take-3 exercise runs (37 notes, model trained on takes 1–2) | 78% | 92% |
| Synthetic let-ring sequences, frets 0–5 at random (480 notes, held-out half) | 89% | 90% |

On the held-out takes, every note with the right pitch also got the right string. The remaining errors are pitch errors, mostly a high-string note read an octave or two low.

## Project layout

```
Guitar Analyzer.bat     one-click start (creates .venv on first run)
launcher.py             start-up window for every tool
run_hmi.py              real-time monitor
run_recorder.py         prompted recorder for labeled clips
record_exercise.py      let-ring exercise recorder (known tab; exercises in guitar_hmi/exercises.py)
import_hf_dataset.py    Hugging Face dataset import
sim_benchmark.py        DSP analyzer vs. simulator ground truth
guitar_hmi/             monitor: analysis, model analyzer, chord fit, fingering, tab, UI, audio sources
Training/               data loading, features, models, training and evaluation scripts
```

**String numbering:** `labels.csv` and the training code use string 1 = high E to 6 = low E. Inside `guitar_hmi`, string index 0 = low E.

## Roadmap

Done: let-ring exercise recorder and scorer, spectral-flux pluck detection, ringing subtraction before the pitch search, and training the string model on notes cut from the exercises.

1. Score a new day's exercise takes with the current model (the strongest test so far), then add them to training.
2. Fix the remaining pitch errors: high-string notes read an octave or two low, and some notes heard as a pitch that is still ringing.
3. More exercises beyond open position, so the string model is also tested where "lowest fret" is the wrong answer.
4. Return to chords, then extend beyond first position.

## Data credits

The Hugging Face clips come from [collegefishiesd/guitar-fretboard-notes](https://huggingface.co/datasets/collegefishiesd/guitar-fretboard-notes) (CC BY-SA 4.0).
