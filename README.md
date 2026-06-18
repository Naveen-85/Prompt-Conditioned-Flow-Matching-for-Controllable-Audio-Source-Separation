# Transformer Based Selective Hearing System for Real-Time Audio Filtering and Source Separation

## Overview

This project presents a **selective hearing system** for audio analysis, filtering, and
source separation. The system identifies important sound events in a mixed audio signal and
isolates the target sound of interest from the mixture. It combines **sound event
classification** with **prompt-conditioned sound isolation** in a unified pipeline.

The application was developed as part of the thesis project *"Transformer Based Selective
Hearing System for Real-Time Audio Filtering and Source Separation."* The core idea is to
simulate selective listening: first understand what sound is present, then separate the
desired sound from overlapping background audio.

The system uses a **two-stage architecture**:

1. **Sound Event Classification** — the input audio is converted into log-mel spectrogram
   features and classified using a knowledge-distilled model. A large **Audio Spectrogram
   Transformer (AST)** acts as the teacher, and a smaller CNN-Transformer student model is
   used for efficient inference.
2. **Prompt-Based Sound Isolation** — the predicted sound class is used as a conditioning
   prompt for the isolation model, which separates the target sound from the mixture using a
   prompt-guided flow-matching generative approach.

---

# Part 1 — Sound Isolation (main contribution)-For Publication purpose 

The primary contribution of this work is **prompt-based sound isolation**: given an audio
mixture and a text prompt (for example `dog barking`, `human speech`, or `music`), the model
isolates and reconstructs only the requested source.

### Method

- The mixture is loaded as mono at 24 kHz and cropped/padded to 6 seconds.
- A frozen **EnCodec** codec encodes the audio into a latent representation.
- A frozen **T5** text encoder turns the prompt into embeddings.
- A **flow-matching Transformer** (self-attention over audio + cross-attention to the prompt)
  learns a velocity field that transports noise toward the target-source latent, conditioned
  on the prompt.
- At inference, the velocity field is integrated over a chosen number of ODE steps, and
  EnCodec decodes the result back into a waveform.

### Where the work is

All of the sound-isolation work — data generation, model, training, inference, evaluation,
baselines, figures, and statistics — is contained in the notebook:

```
Sound_isolation_using _flowmatching_model.ipynb
```

Refer to this notebook for the complete end-to-end implementation. The application
(`app.py`) is the deployable isolation demo extracted from that notebook.

### Experiments and evaluation (what we did)

The notebook contains the full set of experiments carried out to evaluate the isolation
model:

- **Dataset generation.** Synthetic mixtures were built by combining isolated source clips
  across several sound classes, with randomized gains and signal-to-noise ratios and a fixed
  random seed for reproducibility. Each example pairs a mixture, a target source, and a text
  prompt, split into train / validation / test sets.
- **Training.** The flow-matching model was trained with the velocity-matching (MSE)
  objective using AdamW, with validation-based early stopping. Training and validation loss
  curves are recorded.
- **Main evaluation.** The model was evaluated on a held-out test set using standard source
  separation and speech metrics: **SDR, SI-SDR, STOI, and PESQ**.
- **Ablation study.** Model variants were compared to measure the contribution of individual
  components and configuration choices (for example model capacity and conditioning),
  reported as a separate ablation table.
- **Baselines.** The flow-matching model was compared against other separation approaches,
  including a **CLAP + decoder** prompt-conditioned baseline and standard reference
  separators (Conv-TasNet / Demucs).
- **Statistical significance tests.** **Paired t-tests** were run on per-example metrics to
  compare the flow-matching model against the baselines, reporting mean differences, p-values,
  and 95% confidence intervals so the improvements are shown to be statistically meaningful
  rather than due to chance.
- **Robustness.** Performance was measured under additive noise (low-SNR conditions) to test
  stability on degraded input.
- **Efficiency / latency.** Real-time factor (inference speed) and peak GPU memory were
  measured across different numbers of ODE inference steps, characterizing the
  speed-vs-quality trade-off.
- **Open-set / generalization test.** The trained model was evaluated on an external
  benchmark (DCASE 2024 LASS) without adaptation, to report behaviour under domain mismatch
  on unseen prompts and conditions.
- **Figures.** Loss curves, metric comparison bar charts, spectrogram / waveform examples,
  and attention visualizations were generated to support the analysis.

All of the above — including the printed results, tables, and plots — is preserved in the
notebook so the outcomes can be inspected directly.

### How to run the isolation app

You only need to run `app.py` for the sound-isolation part. Make sure these files are in the
same folder:

```
app/
├── app.py              # Streamlit isolation interface
├── flow_separator.py   # T5 + EnCodec + FlowModel, isolate()
├── audio_io.py         # audio load / pad / normalize / save helpers
└── (checkpoints/best.pt referenced one level up, ../checkpoints/best.pt)
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Always launch with `streamlit run app.py`, **not** `python app.py`. Running it with plain
> `python` will not start the web interface and the file uploader will be empty.

Then in the browser:

1. Confirm the **Flow separator checkpoint** path (defaults to `../checkpoints/best.pt`).
2. **Upload** an audio mixture (`.wav`, `.mp3`, or `.m4a`).
3. Choose what to isolate — `dog barking`, `human speech`, `music`, or a custom prompt.
4. Set the number of **inference steps** (more steps = better quality, slower).
5. Click **Run isolation**, then play or **download** the isolated `.wav`.

> Note: the checkpoint (`best.pt`) must match the model dimensions defined in
> `flow_separator.py` (`D_MODEL=192, N_HEADS=3, N_LAYERS=4`). The weights load with
> `strict=False`, so a mismatched checkpoint will load without error but produce noisy output.

---

# Part 2 — Sound Detection (secondary, optional)

If you also want to see the **sound-detection / classification** work (the knowledge-distilled
AST teacher and CNN-Transformer student model), refer to the notebook:

```
Sound_classification and Sound_isolation.ipynb
```

This notebook contains the classification stage that predicts the dominant sound event, which
in the full two-stage system can be used to automatically generate the prompt for the
isolation stage. It is not required to run the sound-isolation app above.

---

## Requirements

Make sure Python is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

```bash
streamlit run app2.py
```

---

## Project Workflow (full two-stage system)

```
Input audio mixture
   → preprocessing
   → sound classification (predicted sound label)   [Part 2]
   → prompt generation
   → prompt-based sound isolation                    [Part 1]
   → output audio
```

For the isolation-only demo, the classification stage is skipped and the prompt is chosen
directly in the app.

## Applications

Assistive listening, smart audio systems, environmental sound understanding, and audio
source-separation research.
