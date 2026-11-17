# Future design choices

A reference for ideas worth revisiting, collected from the papers and repos reviewed while the live monitor is focused on **first-position single notes over ringing strings**. This is not a todo list. Each entry is something to consider when that part of the project is open again.

The live pipeline today is: spectral-flux pluck → subtract the pre-pluck spectrum → DSP pitch → string model at onset + 250 ms → Viterbi fingering → tab. Chords are a separate path (template fit or `ChordNet`) and are paused.

**String numbering in this repo:** `labels.csv` and `Training/` use string 1 = high E. `guitar_hmi/` uses index 0 = low E.

---

## Sources

| Project | What it is | License |
|---|---|---|
| [ErenReyhanlioglu/Guitar-Transcription](https://github.com/ErenReyhanlioglu/Guitar-Transcription) | Cascaded multi-task CNN on GuitarSet; pitch / string-activity / fret-region heads, then tab. Tablature F1 0.78, TDR 94% on held-out players. | See repo `LICENSE` |
| [rickywong04/Neural-Tone-Mapping-and-Processing](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing) (NTMap) | Student TabCNN clone: 9-frame CQT → 6 strings × 21 states. Weak evaluation (random file split). | See repo |
| [Klangio/guitar-strumming-transcription](https://github.com/Klangio/guitar-strumming-transcription) | ISMIR 2025: one event per strum, up/down, 24 chord **names**. Pickup strum F1 98% vs flux 79%; chord-name accuracy 90% on phone. [Paper](https://arxiv.org/abs/2508.07973) | Apache 2.0 |
| [cwitkowitz/guitar-transcription-continuous](https://github.com/cwitkowitz/guitar-transcription-continuous) (FretNet) | ICASSP 2023: TabCNN + continuous pitch deviation (bends/vibrato). [Paper](https://arxiv.org/abs/2212.03023) | MIT |
| [cwitkowitz/guitar-transcription-with-inhibition](https://github.com/cwitkowitz/guitar-transcription-with-inhibition) | SMC 2022: same-string inhibition and pairwise-likelihood tab layer. [Paper](https://arxiv.org/abs/2204.08094) | MIT |
| [GuitarSet](https://guitarset.weebly.com/) | 360 excerpts, 6 players, hexaphonic string labels. [ISMIR 2018](https://archives.ismir.net/ismir2018/paper/000053.pdf) | Dataset, not code |
| TabCNN (Wiggins & Kim, ISMIR 2019) | Ancestor of NTMap and FretNet: CQT patch → per-string fret softmax. [Paper](https://archives.ismir.net/ismir2019/paper/000033.pdf) | — |

Our plug-in points (where a choice would land if adopted):

| Stage | File |
|---|---|
| Pluck detection | `guitar_hmi/analysis.py` (`FluxOnsetDetector`) |
| Pitch + ringing subtraction | `guitar_hmi/analysis.py` (`_analyze_onset`, `_multi_pitch`, `RINGING_SUBTRACT`) |
| String model | `Training/model.py` (`NoteNet`), `Training/features.py`, `Training/predictor.py` |
| Live string/chord apply | `guitar_hmi/model_analyzer.py` |
| Fingering | `guitar_hmi/fingering.py` |
| Chord mix / model / decode | `Training/chords.py`, `guitar_hmi/chord_fit.py` |
| Chord eval | `Training/eval_chords.py` |
| Let-ring eval | `Training/eval_exercises.py` |
| Exercise recording | `record_exercise.py`, `guitar_hmi/exercises.py` |
| Augmentation | `Training/augment.py` |
| Tab export | `guitar_hmi/tab.py`, `guitar_hmi/sheet.py` |

---

## Already in this repo

Do not re-adopt these. They are here so a later reader does not treat them as new.

| Choice | Where |
|---|---|
| Pitch first, then string (not spectrogram → fret in one shot) | `analysis.py` pitch, then `NoteNet` / `FingeringTracker` |
| Spectral-flux onsets with lookahead and sharp-rise location | `FluxOnsetDetector` in `analysis.py` |
| Subtract the pre-pluck spectrum before searching for new pitches | `RINGING_SUBTRACT` in `_analyze_onset` |
| Train the string model on notes cut from let-ring takes | `Training/exercise_clips.py`, `dataset_ringing/` |
| Score found / pitch / string+fret, split by ringing context | `Training/eval_exercises.py` |
| Held-out clip half for synthetic sequences | `Training/data.py` `take_half`, `Training.py --exclude-half` |
| One-note-per-string at decode time (Viterbi, not a trained inhibition layer) | `guitar_hmi/fingering.py` |
| Mixed synthetic strums from single-note clips | `Training/chords.py` `StrumMixer` |
| Per-string fret-or-silent chord head, constrained to playable shapes | `ChordNet` and `decode()` in `Training/chords.py` |
| Small pitch warp of clips (±15 cents), not a full-semitone chord shift | `Training/augment.py` `warp` |

---

## Picking track

### 1. Split errors the way TDR does

**The choice.** Do not report only “right string+fret.” Count separately: missed note, extra note, right pitch on the wrong string, wrong fret on the right string, octave error. Tablature Disambiguation Rate (TDR) is “among pitches that were correct, how often was the string correct.” That is the metric for the open-A vs string-6-fret-5 family of mistakes.

**Why it matters.** On the let-ring exercises, pitch is already higher than placement. A single “79% string+fret” number hides whether the leftover is octave-low pitch or same-pitch wrong string. Eren’s ablation and figures are built around that split.

**Where it lives.**

- Definition and code: [Eren `src/utils/metrics.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/src/utils/metrics.py)
- Per-string confusion: [Eren `src/utils/analyze_errors.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/src/utils/analyze_errors.py), figure `assets/string_confusion.png`
- FretNet’s similar split (“string-dependent” vs “string-agnostic” note F1): [paper Table 1](https://arxiv.org/html/2212.03023v2), code under `guitar_transcription_continuous/metrics/` and `guitar_transcription_continuous/evaluators/`

**Where it would go here.** `Training/eval_exercises.py` already reports found / pitch / string+fret and ringing context. Add TDR (string correct | pitch correct) and an octave-error bucket. Live debugging of a take is the same idea as the temporary `explain_take.py` notes: print DSP pitch, model string, and whether anything was still ringing.

---

### 2. Auxiliary heads as training losses, not extra inputs

**The choice.** Train extra outputs (which strings are sounding, coarse fret region: open / 1–5 / 6–10 / …, pitch class) **only as extra losses**. Do not concatenate those probabilities into the tab decoder at inference. Eren’s ablation: supervising the auxiliaries helped; feeding them forward as features did not (`configs/ablations/E_fusion_no_auxloss.yaml` vs the fusion configs).

**Why it matters.** If the string model stays a small CNN but starts missing same-pitch placements again, a “which strings could be active?” or “which fret region?” loss is the lever they found, not a bigger concatenated input. The string-activity head is the one that actually describes the instrument when several strings can sound.

**Where it lives.**

- Architecture: [Eren `src/models/cnn_mtl.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/src/models/cnn_mtl.py)
- Losses (focal + uncertainty-weighted multi-task): [Eren `src/utils/losses.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/src/utils/losses.py)
- Ablations: [`configs/ablations/D_no_string_activity.yaml`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/configs/ablations/D_no_string_activity.yaml), [`C_no_fret_region.yaml`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/configs/ablations/C_no_fret_region.yaml), [`E_fusion_no_auxloss.yaml`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/configs/ablations/E_fusion_no_auxloss.yaml)
- Proposed config: [`configs/cnn_mtl_config.yaml`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/configs/cnn_mtl_config.yaml)

**Where it would go here.** Extra heads on `NoteNet` in `Training/model.py`, trained in `Training/Training.py`. Inference in `Training/predictor.py` / `model_analyzer.py` would still take pitch from DSP (or the pitch head) and string logits; it would **not** splice auxiliary probabilities into the final layer.

**Do not do this** as a way to replace flux + DSP. Their whole CNN is offline, ~6.7M parameters, 19-frame windows that look into the future.

---

### 3. Harmonic CQT (or stacked harmonic channels) as a feature

**The choice.** Instead of one CQT or one log-band spectrogram, stack several CQTs whose base frequencies are the first sub-harmonic and the first few harmonics of E2 (or of the guessed F0). Convolution then sees harmonic energy as extra channels. Eren and FretNet both report this beating a plain CQT. FretNet ablation (8) “CQT Features” drops tablature F1 from 0.727 to 0.700.

**Why it matters.** Remaining picking errors are mostly octave and two-octave (high E heard as low E). We already measure harmonic peaks of the guessed F0 in `Training/features.py` (`_harmonic_peaks`, 16 harmonics, inharmonicity bound `MAX_B`). HCQT is the spectrogram version of that idea: give the CNN the harmonic stack, not only a 1-D peak list. Try this if octave errors survive more held-out exercise takes.

**Where it lives.**

- Eren feature build: [`extract_features.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/extract_features.py); HCQT vs CQT ablations under `configs/ablations/hcqt_*.yaml` and `cqt_*.yaml`
- FretNet description: paper §2.1; model input in [`guitar_transcription_continuous/models/fretnet.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/fretnet.py)
- Original harmonic-CQT motivation: Bittner et al., Deep Salience (ISMIR 2017)

**Where it would go here.** `Training/features.py` `spectrogram()` today returns `(batch, 1, bands, frames)`. A harmonic stack would be extra channels in dim 1. `NoteNet`’s first conv already takes `c_in`; change 1 → N channels. Live path would need the same transform in `Training/predictor.py` so training and the monitor match.

---

### 4. Same-string inhibition inside the model

**The choice.** Add a loss (and/or a structured output layer) that punishes two different frets on the **same** string in one frame. FretNet uses \(L_{inh}\) = sum of products of same-string activations, weight \(\lambda = 10\). The earlier inhibition paper also learns a pairwise-likelihood matrix from symbolic tab (DadaGP / GuitarSet) so unlikely fret pairs are suppressed.

**Why it matters for us, and why it does not.** The Viterbi tracker already enforces one placement per note and prefers a playable hand. Inhibition is the **training-time** version of that constraint, useful if we ever emit a **frame-level** 6 × (fret or silent) map instead of one event per pluck. It does **not** fix open A (string 5 fret 0) vs string 6 fret 5: those are different strings, same pitch.

**Where it lives.**

- Inhibition matrices and utilities: [`guitar_transcription_inhibition/inhibition/inhibition_matrix_utils.py`](https://github.com/cwitkowitz/guitar-transcription-with-inhibition/blob/main/guitar_transcription_inhibition/inhibition/inhibition_matrix_utils.py)
- Tab output layer: [`guitar_transcription_inhibition/models/tablature_layers.py`](https://github.com/cwitkowitz/guitar-transcription-with-inhibition/blob/main/guitar_transcription_inhibition/models/tablature_layers.py)
- TabCNN variants that use it: [`guitar_transcription_inhibition/models/tabcnn_variants.py`](https://github.com/cwitkowitz/guitar-transcription-with-inhibition/blob/main/guitar_transcription_inhibition/models/tabcnn_variants.py)
- FretNet loss (paper eq. 1, ablation “No Inhibition”): [`guitar_transcription_continuous/models/fretnet.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/fretnet.py)

**Where it would go here.** Only if `ChordNet` or a future frame-level tab head starts firing two frets on one string. Then add the loss next to `Training/chords.py` / `Training/Training.py`. Do not duplicate it in `fingering.py`, which already forbids that at decode time.

---

### 5. Learned onset detector (only if flux starts failing)

**The choice.** Replace or gate spectral flux with a learned onset or “time-to-onset” head. Klangio’s regression-based onset (a distribution over time-to-next-strum) beat flux by a wide margin **on strums** (F1 98% vs 79% on pickup). FretNet’s onset head (Onsets-and-Frames style) raised **note precision**; removing it raised F1 only by recalling extra notes — the same extra-trigger problem flux showed on isolated clips.

**Why it matters.** Flux already finds 99% of plucks on the let-ring exercises, including soft ones over ringing. A learned detector is for **Chords mode** (one swipe is many strings over 20–80 ms, so flux can split one strum into several “notes”) or if picking extras become a real live problem.

**Where it lives.**

- Klangio CRNN + onset: [`klangio/modules/strumming_crnn/model.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/klangio/modules/strumming_crnn/model.py), training [`scripts/train_phone.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/train_phone.py), pretrained [`checkpoints/wc/step=19000-f1=0.8225.ckpt`](https://github.com/Klangio/guitar-strumming-transcription/tree/main/checkpoints/wc)
- FretNet onset head: paper §2.4; [`fretnet.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/fretnet.py)
- Original onset+frame recipe: Hawthorne et al., Onsets and Frames (ISMIR 2018)

**Where it would go here.** A second detector beside `FluxOnsetDetector` in `analysis.py`, selected by monitor mode (Picking vs Chords), or a “merge onsets within 80 ms into one strum” gate in `model_analyzer.py` before chord fit. Do not replace flux for picking unless extras/misses show up on new held-out takes.

---

## Chord track

### 6. One event per strum, not six string onsets

**The choice.** Treat a swipe as a single timed event with a chord (and optionally direction). Note-tracking models split one strum into several notes because they look for single-string attacks. Klangio state this as the reason picking models fail on strums. We already chose this at the product level (monitor modes Picking / Chords / Auto).

**Why it matters.** When Chords mode comes back, the failure to watch is flux (or the auto-mode pitch count) emitting several events during one swipe. Fix grouping before replacing the string model.

**Where it lives.**

- Paper: [arXiv:2508.07973](https://arxiv.org/abs/2508.07973)
- Model that jointly does onset + direction + name: [`klangio/modules/strumming_crnn/model.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/klangio/modules/strumming_crnn/model.py)
- Inference notebook: [`scripts/inference.ipynb`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/inference.ipynb)

**Where it would go here.** `DspAnalyzer` / `ModelAnalyzer` chord branch, and the Auto mode decision in `guitar_hmi/analysis.py` / `model_analyzer.py`. `Training/eval_chords.py` should score “was it one strum?” separately from “was the shape right?”

---

### 7. Scripted chord (and strum) recording, same idea as let-ring exercises

**The choice.** Players follow a written plan (tempo, pattern, chord sequence). Onsets are proposed by flux (and, in Klangio, a metronome grid and a watch IMU), then a person checks. Labels come from the plan, not from listening after the fact.

**Why it matters.** This is exactly `record_exercise.py` for picking. Chords need the same thing: known shapes in known order, with ringing, so `eval_chords.py` can score real takes the way `eval_exercises.py` scores picking. Klangio’s 90 minutes of real pickup + mic only became usable because of the plan.

**Where it lives.**

- Dataset files: `dataset/klangio-gst-mm-2025/recording_*.strums` plus `*_line.wav` (pickup) and `*_phone.wav` (mic) in the Klangio repo
- Paper § on semi-automatic annotation; README “Multimodal dataset” / “Semi-automatic annotation”
- Our picking version: `record_exercise.py`, `guitar_hmi/exercises.py`, `Training/eval_exercises.py`

**Where it would go here.** New entries in `guitar_hmi/exercises.py` (or a `chords` list), a recorder path that saves shape + take like the exercise JSON, and a real-strum folder consumed by `Training/eval_chords.py` (`real_strums` already reads labeled chords from `labels.csv`).

---

### 8. Synthetic strums first, then fine-tune on real ones

**The choice.** Train on hours of mixed/synthesized strums, then fine-tune on a smaller set of real pickup (and optionally mic) recordings. Klangio: synthetic data was enough to **detect** strums; **chord names** and **up/down** needed real audio. Their published checkpoint is pretrained on synthetic, then fine-tuned on phone + line.

**Why it matters.** We already mix clips in `StrumMixer`. The missing half is: keep a held-out set of **real** strums, train on mix, then fine-tune, and never score the mix as if it were the guitar.

**Where it lives.**

- Klangio training entry: [`scripts/train_phone.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/train_phone.py), data prep [`scripts/prepare_data_for_training.ipynb`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/prepare_data_for_training.ipynb)
- Checkpoint: `checkpoints/wc/step=19000-f1=0.8225.ckpt`
- Our mixer: `Training/chords.py` `StrumMixer.strum`; training `Training/train_chords.py`; eval `Training/eval_chords.py` (piezo half-split and `real strums` fold)

---

### 9. Pitch-shift the whole strum by ±6 semitones, and shift the label

**The choice.** For chord/strum training, transpose the audio by whole semitones (Klangio: ±6) and change the chord label to match. This is not the ±15 cent `warp()` used for single notes (that must stay inside the labeled semitone). Klangio report +14 F1 on up-strums and +10 on down-strums from this one change.

**Why it matters.** Chord names and shapes are sparse. Shifting a C shape to D is free extra data if the label moves with it. Useless for the current string model if the label is a specific string+fret clip.

**Where it lives.**

- Klangio: [`klangio/modules/strumming_crnn/augmentation.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/klangio/modules/strumming_crnn/augmentation.py)
- Paper “Robust generalization” bullet in the Klangio README

**Where it would go here.** A chord-only path in `Training/augment.py` or inside `StrumMixer` / `train_chords.py`. Must remap voicing MIDI and the 24-name (or fret) target. Do not turn this on in `Training.py` for single notes.

---

### 10. Drop a string in mixed strums

**The choice.** In about half of synthetic strums, mute the last string of the swipe (Klangio: amateur / incomplete strum). The model then has to allow a missing string in a known shape.

**Why it matters.** `StrumMixer` currently plays every string in the chosen shape (`strum()` loops `order` of all keys in `frets`). Real players miss the high E or don’t finish a down-strum. `ChordFitter` already has `SILENT_SHARE` to drop a string that explains almost none of the spectrum; the **training** mixer does not randomly omit one.

**Where it lives.**

- Described in the Klangio paper (synthetic imperfections); audio FX also in [`klangio/modules/strumming_crnn/augmentation.py`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/klangio/modules/strumming_crnn/augmentation.py)
- Our mixer: `Training/chords.py` `StrumMixer.strum` (delay + gain per string, no drop)
- Our fitter silence rule: `guitar_hmi/chord_fit.py` `SILENT_SHARE`

**Where it would go here.** Inside `StrumMixer.strum`: with probability *p*, drop the last string in `order` (or a random interior string) and set that target class to “not sounding” (`targets_of`).

---

### 11. Score chord **name** and exact voicing separately

**The choice.** Klangio’s 90% is 24 class names (C, Am, …), not `x32010`. A correct C with the wrong inversion still counts. Our `eval_chords.py` scores exact fret dicts (`score_shapes`). Both numbers are useful; they are not interchangeable.

**Why it matters.** When comparing to Klangio or to a pretrained checkpoint, do not treat 90% name accuracy as 90% tab. When improving `ChordNet`, a name-level score tells you whether the harmony is right even when the voicing is not.

**Where it lives.**

- Klangio metrics / eval: [`scripts/evaluate.ipynb`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/evaluate.ipynb), model head in `model.py`
- Our exact-shape score: `Training/chords.py` `score_shapes`, used by `Training/eval_chords.py`
- Name from a shape: `guitar_hmi/guitar.py` `name_chord`, `model_analyzer.py` `chord_name`

**Where it would go here.** Extra column in `eval_chords.py`: map predicted and target fret dicts through `name_chord` / `VOICINGS` and report name accuracy beside exact-shape accuracy.

---

### 12. Six independent “fret or mute” outputs (frame or strum)

**The choice.** One classifier per string, each with classes {silent, fret 0, fret 1, …}. TabCNN, NTMap, FretNet’s tab head, and our `ChordNet` all use this shape. The picking `NoteNet` does **not**: it picks a pitch, then a string that can play it.

**Why it matters.** This is the right head as soon as several strings sound at once (chords, fingerpicking). NTMap is otherwise not worth copying; this output layout is the piece to keep.

**Where it lives.**

- NTMap: [`model_cnn.py`](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing/blob/main/model_cnn.py) (6 × 21), [`generate_tabs.py`](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing/blob/main/generate_tabs.py)
- TabCNN paper: [ISMIR 2019](https://archives.ismir.net/ismir2019/paper/000033.pdf)
- FretNet tab head size \(6(F+2)\) including silence: paper §2.2, [`fretnet.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/fretnet.py)
- Ours already: `Training/chords.py` `ChordNet` (`N_CLASSES = MAX_FRET + 2`, 0 = not sounding)

**Where it would go here.** Already the chord path. For fingerpicking (several strings, not a strum), reuse this head rather than calling `NoteNet` six times.

---

### 13. Pretrained Klangio checkpoint as a chord-**name** baseline only

**The choice.** Run their Apache-2.0 CRNN on a recorded strum and compare **names** to `name_chord` of our shape. It cannot write a fretboard. 10-second windows, not a 250 ms live path.

**Where it lives.** [`checkpoints/wc/`](https://github.com/Klangio/guitar-strumming-transcription/tree/main/checkpoints/wc), [`scripts/inference.ipynb`](https://github.com/Klangio/guitar-strumming-transcription/blob/main/scripts/inference.ipynb)

**Where it would go here.** A one-off comparison script next to `Training/eval_chords.py`, not inside `ModelAnalyzer`.

---

## Later: technique, data, export

### 14. Continuous pitch deviation (bends, vibrato, slides)

**The choice.** For each active string+fret, predict a continuous offset in semitones from the nominal fret pitch (FretNet: one extra neuron per pair, Continuous Bernoulli, scaled to \([-r, r]\); they used \(r = 1\) because GuitarSet barely annotates technique). Discrete activity stays on the fret; the contour rides on top.

**Why it matters.** Integer-fret tab cannot mark `b`, `~`, or slides. Only worth it after placement is solid. Does not help open-A vs s6f5.

**Where it lives.**

- Layer: [`guitar_transcription_continuous/models/continuous_layers.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/continuous_layers.py)
- Model: [`fretnet.py`](https://github.com/cwitkowitz/guitar-transcription-continuous/blob/main/guitar_transcription_continuous/models/fretnet.py)
- Paper §2.3 and Figure 1; training in `six_fold_cv_scripts/experiment.py`
- Downstream technique-from-contour (cited by FretNet): Su et al., TENT (TISMIR 2019)

**Where it would go here.** A new head beside `NoteNet` or on a frame model; tab marks in `guitar_hmi/tab.py` / `sheet.py`. Needs labeled contours (GuitarSet pitch tracks, or new recordings with known bends).

---

### 15. GuitarSet as a **test** set, not as training data

**The choice.** Six players, hexaphonic ground truth, full fretboard, mix of styles. Excellent for “does this idea work on someone else’s guitar?” Dangerous as training if the live input is still the UM2 piezo: their mix/mic/hex-debleeded audio does not sound like one contact pickup.

**Where it lives.** [guitarset.weebly.com](https://guitarset.weebly.com/); Eren ingest [`src/data_loader.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/src/data_loader.py) + [`extract_features.py`](https://github.com/ErenReyhanlioglu/Guitar-Transcription/blob/main/extract_features.py); FretNet `guitar_transcription_continuous/datasets/`; NTMap [`parse_jams.py`](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing/blob/main/parse_jams.py) (they train on **debleeded hex**, which makes string identity easier than a single pickup).

**Where it would go here.** A later eval folder, scored with the same found / pitch / TDR split. Do not mix GuitarSet into `Training.py` while the monitor is piezo-only. Six-fold **player** splits, never a random file split (NTMap’s 80/10/10 lets the same player into train and test).

---

### 16. MIDI export from the tab

**The choice.** NTMap writes MIDI from the 6 × fret frame dump. We already write ASCII, MusicXML, and PDF.

**Where it lives.** [NTMap `generate_tabs.py`](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing/blob/main/generate_tabs.py)

**Where it would go here.** `TabSession` export in `guitar_hmi/tab.py`, next to the MusicXML writer. Optional; DAW playback only.

---

### 17. Stem splitting (Demucs) for full-band audio

**The choice.** NTMap can split a mix before tabbing. The live monitor is already a guitar-only piezo.

**Where it lives.** [NTMap `split_audio.py`](https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing/blob/main/split_audio.py)

**Skip** unless the project starts tabbing songs off YouTube/mic mixes.

---

### 18. Up / down strum as a tab mark

**The choice.** Klangio classifies plectrum direction. Useful for a strumming diagram. Not needed for string+fret tab.

**Where it lives.** Same CRNN head as §6; labels in `recording_*.strums`.

**Skip** unless notation should show \(\sqcap\) / \(\sqcup\).

---

### 19. Smartwatch / IMU for labeling only

**The choice.** Klangio used an ESP32 watch to help **annotate** direction. Inference is audio-only.

**Skip** for this project. `record_exercise.py` plus a written chord plan is the equivalent.

---

## Explicitly not worth copying as the live engine

These models are useful as **ideas** above. Dropping them in as the monitor would fight the current design (pluck → 250 ms → small net, piezo, frets 0–5).

| Model | Why not as the live engine |
|---|---|
| Eren CNN | Offline, looks ahead ~200 ms, GuitarSet mix, 0–19 frets, 6.7M params |
| NTMap / TabCNN | No pluck detector; frame tab repeats a fret for the whole ring; trained on hex-debleeded GuitarSet; random split |
| FretNet | Same TabCNN family; headline tab F1 0.727 on GuitarSet; continuous head is for bends |
| Klangio CRNN | Chord **names**, 10 s windows, no voicing |

---

## Suggested order when a track reopens

1. **Still picking:** TDR / octave buckets in `eval_exercises.py` (§1). If octave errors remain, harmonic-stack features (§3). Auxiliary **losses** only if placement (same pitch, wrong string) returns (§2).
2. **Chords:** one-event grouping (§6), scripted real strums (§7), drop-a-string + ±6 shift in the mixer (§10, §9), name vs voicing scores (§11), then synthetic→real fine-tune (§8). Learned strum onset (§5) only if Chords mode double-triggers.
3. **After integer frets are solid:** deviation head (§14), GuitarSet as a held-out test (§15), MIDI if anyone asks (§16).
