# BoneGAN: Generative Adversarial Networks for Bone-Conducted Speech Enhancement

BoneGAN is a conditional Generative Adversarial Network (cGAN) designed to restore and enhance bone-conducted speech. Bone-conducted microphones capture speech vibrations transmitted through cranial bones and facial tissue; while resistant to ambient acoustic noise, bone-conducted speech suffers from severe attenuation of higher frequencies (typically above 1 kHz), resulting in muffled and low-intelligibility audio. 

BoneGAN models the mapping from bone-conducted spectrograms to high-fidelity air-conducted speech spectrograms using a pix2pix architecture with paired adversarial and L1 reconstruction losses.

---

## Attribution

> **Generator and discriminator adapted from the TensorFlow pix2pix tutorial; adapted to paired bone/air-conducted speech spectrograms.**

---

## Leakage-Safe Data Splitting

Previous implementations partitioned speech datasets by shuffling 2D/3D spectrogram time slices across the entire recording pool. Because adjacent spectrogram frames from the same utterance share acoustic characteristics, speaker traits, and background noise, randomly splitting frames results in significant **data leakage** between training and test sets.

BoneGAN enforces **leakage-safe partitioning**:
- Data splitting is performed strictly by **recording clip** (or by **speaker** when speaker identifiers are present).
- All spectrogram time slices belonging to a given utterance remain confined to a single partition.
- Train / Validation / Test partitions (default 70% / 15% / 15%) are computed and saved to reproducible file lists:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
- The evaluation pipeline strictly processes clips listed in `splits/test.txt`.

---

## Reproducibility and Determinism

Reproducibility is enforced from a central configuration:
- A single `seed` parameter (default `42`) is set across Python's `random`, `numpy.random`, and TensorFlow (`tf.random.set_seed`, `tf.keras.utils.set_random_seed`).
- `PYTHONHASHSEED` is set in the environment.
- Deterministic operations are enabled via `tf.config.experimental.enable_op_determinism()`.
- **Note on Determinism:** On CPU execution, operations are strictly deterministic. On certain GPU environments with CUDA/cuDNN, specific atomic reduction kernels or historical `Conv2DTranspose` backward paths may lack deterministic implementations or may run with reduced performance when determinism is enforced.
- Every training run automatically logs a copy of the active `config.yaml` inside the checkpoint directory for full provenance.

---

## Repository Structure

```
├── Bone_GAN.ipynb           # Lightweight demo notebook importing pipeline modules
├── config.yaml              # Centralized configuration (hyperparameters, STFT, paths)
├── data.py                  # STFT processing, pairing, waveform inversion, and split creation
├── evaluate.py              # Test-split evaluation (PESQ, STOI, LSD)
├── losses.py                # Adversarial BCE and L1 generator/discriminator losses
├── model.py                 # Pix2pix U-Net Generator and PatchGAN Discriminator
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Pinned dependency versions
├── tests/
│   └── test_pipeline.py     # End-to-end smoke test using synthetic audio (<1 min)
├── train.py                 # GradientTape training loop with checkpointing and logging
└── utils.py                 # Reproducibility seed setting and configuration helpers
```

---

## Installation

Ensure you have Python 3.10 or 3.11 installed.

```bash
# Clone the repository
git clone https://github.com/Arifuzzamanjoy/BoneGAN-Generative-Adversarial-Networks-for-Bone-Conducted-Speech.git
cd BoneGAN-Generative-Adversarial-Networks-for-Bone-Conducted-Speech

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Quickstart & Smoke Test

You can verify the entire pipeline end-to-end in under a minute without needing access to the private dataset. The smoke test synthesizes paired harmonic speech signals, applies low-pass filtering to simulate bone conduction, creates leakage-safe splits, trains for 1 epoch, and verifies PESQ/STOI/LSD evaluation:

```bash
pytest tests/test_pipeline.py
```

---

## Usage Workflow

### 1. Data Preparation & Splitting

Place your paired `.wav` files into your data directories (configurable in `config.yaml`):
- `data/clean/`: Target air-conducted speech recordings.
- `data/noisy/`: Paired bone-conducted speech recordings.

Generate the leakage-safe train/val/test splits:
```bash
python data.py --create-splits --config config.yaml
```
This generates `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.

### 2. Training

Train the model using the centralized configuration:
```bash
python train.py --config config.yaml
```

Optional CLI overrides:
```bash
python train.py --config config.yaml --epochs 50 --batch_size 1 --checkpoint_dir training_checkpoints
```

Training logs per-epoch generator total loss, GAN adversarial loss, L1 reconstruction loss, and discriminator loss to both stdout and TensorBoard (`logs/`).

### 3. Evaluation

Evaluate the trained model strictly on the held-out test split:
```bash
python evaluate.py --checkpoint training_checkpoints --config config.yaml
```

This reconstructs the audio waveforms and computes:
- **PESQ (Perceptual Evaluation of Speech Quality):** Wideband ITU-T P.862.2 score.
- **STOI (Short-Time Objective Intelligibility):** Correlation score between clean and enhanced speech.
- **LSD (Log-Spectral Distance):** Root-mean-square spectral envelope difference in dB.

Results are printed to the console and written to `results/metrics.json` with both mean and sample standard deviation across test clips:

```json
{
  "pesq": {
    "mean": 0.0,
    "std": 0.0
  },
  "stoi": {
    "mean": 0.0,
    "std": 0.0
  },
  "lsd": {
    "mean": 0.0,
    "std": 0.0
  },
  "num_clips": 0,
  "clips": []
}
```

Add `--save-audio` to export the enhanced `.wav` audio files to the results folder.

---

## Results

Because the bone-conducted speech dataset is private and not distributed with this repository, metric numbers are not hardcoded or fabricated in this document. Numerical results (PESQ, STOI, and LSD) are produced directly by executing `evaluate.py` on the test split after training on the real dataset.

---

## Limitations

1. **Small Dataset Size:** The private paired bone/air corpus contains limited vocabulary and speaker variety, which may constrain out-of-distribution generalization.
2. **Single Acoustic & Sensor Domain:** Bone conduction transfer functions depend heavily on transducer placement (e.g., mastoid, forehead, or jaw), skin contact pressure, skull anatomy, and sensor frequency responses. Performance may vary across different sensor hardware.
3. **Magnitude-Only Spectrogram Mapping:** The current model reconstructs magnitude spectrograms and relies on bone phase transfer or Griffin-Lim for waveform inversion. Joint complex spectrogram modeling or neural vocoders (e.g., HiFi-GAN) could further reduce phase artifacts.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
