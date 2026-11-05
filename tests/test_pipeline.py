"""
End-to-end smoke test for BoneGAN pipeline using synthetic signals.
Runs in under a minute without requiring the private dataset.
"""

import json
import os
import shutil
import sys
import tempfile
import numpy as np
import pytest
from scipy import signal
import soundfile as sf
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import create_dataset_from_files, create_splits, load_split_file_list
from evaluate import evaluate
from train import train
from utils import set_seed


def generate_synthetic_audio(
    duration: float = 1.0,
    sr: int = 16000,
    base_freq: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic speech-like pair:
    - clean: harmonic multi-tone signal (300Hz, 800Hz, 2400Hz, 4000Hz).
    - bone/noisy: low-pass filtered version (simulating tissue attenuation >1kHz).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    clean = (
        0.5 * np.sin(2 * np.pi * base_freq * t)
        + 0.3 * np.sin(2 * np.pi * (base_freq * 2.5) * t)
        + 0.15 * np.sin(2 * np.pi * 2400.0 * t)
        + 0.1 * np.sin(2 * np.pi * 4000.0 * t)
    )

    # Low-pass filter (Butterworth 4th order at 1000 Hz) to simulate bone-conduction attenuation
    sos = signal.butter(4, 1000.0, btype="lowpass", fs=sr, output="sos")
    bone = signal.sosfilt(sos, clean)
    bone = bone + 0.01 * np.random.randn(len(bone))

    clean = (clean / (np.max(np.abs(clean)) + 1e-6)).astype(np.float32) * 0.9
    bone = (bone / (np.max(np.abs(bone)) + 1e-6)).astype(np.float32) * 0.9

    return clean, bone


def test_end_to_end_pipeline():
    """
    Tests complete pipeline: data generation, leakage-safe splits,
    dataset creation, training for 1 epoch, and evaluation with PESQ, STOI, and LSD.
    """
    set_seed(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        clean_dir = os.path.join(tmp_dir, "clean")
        noisy_dir = os.path.join(tmp_dir, "noisy")
        splits_dir = os.path.join(tmp_dir, "splits")
        ckpt_dir = os.path.join(tmp_dir, "checkpoints")
        results_dir = os.path.join(tmp_dir, "results")
        log_dir = os.path.join(tmp_dir, "logs")

        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        # 1. Create 5 synthetic paired audio clips
        sr = 16000
        for i in range(5):
            clean_audio, bone_audio = generate_synthetic_audio(
                duration=1.0, sr=sr, base_freq=250.0 + i * 50.0
            )
            sf.write(os.path.join(clean_dir, f"clip_{i:02d}.wav"), clean_audio, sr)
            sf.write(os.path.join(noisy_dir, f"clip_{i:02d}.wav"), bone_audio, sr)

        # 2. Test leakage-safe split creation
        splits = create_splits(
            clean_folder=clean_dir,
            noise_folder=noisy_dir,
            splits_dir=splits_dir,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            seed=42,
            split_by="recording",
        )

        train_files = {p[0] for p in splits["train"]}
        val_files = {p[0] for p in splits["val"]}
        test_files = {p[0] for p in splits["test"]}

        # Verify strict 0% overlap between splits
        assert len(train_files.intersection(val_files)) == 0, "Train and Val overlap!"
        assert len(train_files.intersection(test_files)) == 0, "Train and Test overlap!"
        assert len(val_files.intersection(test_files)) == 0, "Val and Test overlap!"
        assert len(test_files) > 0, "Test set should not be empty!"

        # 3. Test dataset construction
        train_ds = create_dataset_from_files(
            clean_folder=clean_dir,
            noise_folder=noisy_dir,
            file_list=splits["train"],
            batch_size=1,
            is_training=True,
            seed=42,
        )

        batch_count = 0
        for x_b, y_b in train_ds:
            assert x_b.shape == (1, 256, 256, 3)
            assert y_b.shape == (1, 256, 256, 3)
            batch_count += 1
        assert batch_count > 0

        # 4. Prepare temporary config.yaml
        config = {
            "seed": 42,
            "stft": {
                "sr": 16000,
                "window": "hann",
                "nfft": 256,
                "noverlap": 192,
                "reps_horizontal": 8,
                "reps_vertical": 2,
                "slice_time_frames": 32,
            },
            "model": {
                "img_height": 256,
                "img_width": 256,
                "input_channels": 3,
                "output_channels": 3,
            },
            "loss": {
                "lambda_l1": 100.0,
            },
            "optimizer": {
                "generator_lr": 0.0002,
                "discriminator_lr": 0.0002,
                "beta_1": 0.5,
                "beta_2": 0.999,
            },
            "data": {
                "clean_folder": clean_dir,
                "noise_folder": noisy_dir,
                "splits_dir": splits_dir,
                "train_ratio": 0.60,
                "val_ratio": 0.20,
                "test_ratio": 0.20,
                "split_by": "recording",
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "buffer_size": 10,
                "checkpoint_dir": ckpt_dir,
                "checkpoint_prefix": "ckpt",
                "log_dir": log_dir,
                "save_every_epochs": 1,
            },
            "evaluation": {
                "results_dir": results_dir,
                "metrics_file": os.path.join(results_dir, "metrics.json"),
            },
        }

        config_path = os.path.join(tmp_dir, "test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        # 5. Run 1-epoch training
        saved_ckpt = train(config_path=config_path, epochs=1)
        assert os.path.exists(ckpt_dir)
        assert os.path.exists(os.path.join(ckpt_dir, "config.yaml")), "Config was not logged next to run!"

        # 6. Run evaluation on test split
        metrics = evaluate(
            checkpoint_path=ckpt_dir,
            config_path=config_path,
            save_audio=True,
            output_dir=results_dir,
        )

        # 7. Check metrics output
        assert os.path.exists(os.path.join(results_dir, "metrics.json"))
        with open(os.path.join(results_dir, "metrics.json"), "r") as f:
            saved_metrics = json.load(f)

        assert "pesq" in saved_metrics
        assert "stoi" in saved_metrics
        assert "lsd" in saved_metrics
        assert saved_metrics["num_clips"] == len(splits["test"])
        assert not np.isnan(saved_metrics["stoi"]["mean"])
        assert not np.isnan(saved_metrics["lsd"]["mean"])
