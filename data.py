"""
Data loading, STFT processing, pairing, and leakage-safe dataset splitting for BoneGAN.
"""

import os
import re
import random
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from natsort import natsorted
from scipy import signal
import soundfile as sf
import tensorflow as tf

from utils import set_seed, load_config


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples to target sample rate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate


def compute_stft(
    audio: np.ndarray,
    sr: int = 16000,
    nfft: int = 256,
    noverlap: int = 192,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes STFT of an audio signal using scipy.signal.stft.
    Returns (f, t, Zxx).
    """
    win = signal.windows.hann(nfft) if window == "hann" else signal.get_window(window, nfft)
    f, t, Zxx = signal.stft(audio, sr, window=win, nperseg=nfft, noverlap=noverlap)
    return f, t, Zxx


def extract_slices(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    sr: int = 16000,
    nfft: int = 256,
    noverlap: int = 192,
    window: str = "hann",
    reps_horizontal: int = 8,
    reps_vertical: int = 2,
    slice_time_frames: int = 32,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Processes a pair of clean and bone/noisy audio signals into vertical/horizontal
    expanded STFT magnitude slices.

    Returns:
        (slices_n, slices_c, phase_n, phase_c)
    """
    # Align lengths
    min_len = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_len]
    noise_audio = noise_audio[:min_len]

    # Compute STFT
    _, _, Zxx_noisy = compute_stft(noise_audio, sr, nfft, noverlap, window)
    _, _, Zxx_clean = compute_stft(clean_audio, sr, nfft, noverlap, window)

    phase_noisy = np.angle(Zxx_noisy)
    phase_clean = np.angle(Zxx_clean)

    mag_clean = np.abs(Zxx_clean)
    mag_noisy = np.abs(Zxx_noisy)

    n_time = mag_clean.shape[1]
    n_slices = n_time // slice_time_frames

    slices_c_list = []
    slices_n_list = []

    for i in range(n_slices):
        slice_n = mag_noisy[:, i * slice_time_frames : (i + 1) * slice_time_frames]
        slice_c = mag_clean[:, i * slice_time_frames : (i + 1) * slice_time_frames]

        if slice_n.shape[1] < slice_time_frames:
            pad_width = slice_time_frames - slice_n.shape[1]
            slice_n = np.pad(slice_n, ((0, 0), (0, pad_width)))
            slice_c = np.pad(slice_c, ((0, 0), (0, pad_width)))

        # Repeat along horizontal axis (time frames)
        slice_c_horizontal = np.repeat(slice_c, reps_horizontal, axis=1)
        slice_n_horizontal = np.repeat(slice_n, reps_horizontal, axis=1)

        # Concatenate along vertical axis (frequency bins)
        # Notebook logic: slice_c_vertical = np.concatenate((slice_c_horizontal, slice_c_horizontal))
        if reps_vertical == 2:
            slice_c_vertical = np.concatenate((slice_c_horizontal, slice_c_horizontal), axis=0)
            slice_n_vertical = np.concatenate((slice_n_horizontal, slice_n_horizontal), axis=0)
        else:
            slice_c_vertical = np.tile(slice_c_horizontal, (reps_vertical, 1))
            slice_n_vertical = np.tile(slice_n_horizontal, (reps_vertical, 1))

        slices_c_list.append(slice_c_vertical)
        slices_n_list.append(slice_n_vertical)

    # If audio is very short and yields < 3 slices, pad with the last slice so at least one 3D image can be formed
    while len(slices_c_list) < 3 and len(slices_c_list) > 0:
        slices_c_list.append(slices_c_list[-1])
        slices_n_list.append(slices_n_list[-1])

    return slices_c_list, slices_n_list, phase_noisy, phase_clean


def prepare_3d_images(
    slices_c: List[np.ndarray],
    slices_n: List[np.ndarray],
    img_height: int = 256,
    img_width: int = 256,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Stacks consecutive 3 slices along channel axis into (258, 256, 3), resizes to
    (img_height, img_width), and applies per_image_standardization.

    Returns:
        inputs (noisy/bone 3D images), targets (clean 3D images),
        stats_noisy [(mean, std), ...], stats_clean [(mean, std), ...]
    """
    images_c_3d = []
    images_n_3d = []
    stats_c = []
    stats_n = []

    for i in range(0, len(slices_c) - 2, 3):
        img_c = np.dstack([slices_c[i], slices_c[i + 1], slices_c[i + 2]])
        img_n = np.dstack([slices_n[i], slices_n[i + 1], slices_n[i + 2]])

        # Resize to (256, 256)
        img_c_resized = tf.image.resize(
            img_c, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ).numpy()
        img_n_resized = tf.image.resize(
            img_n, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ).numpy()

        # Compute mean and std before standardization for potential inversion
        mean_c, std_c = float(np.mean(img_c_resized)), float(np.std(img_c_resized))
        mean_n, std_n = float(np.mean(img_n_resized)), float(np.std(img_n_resized))
        stats_c.append((mean_c, std_c))
        stats_n.append((mean_n, std_n))

        # Standardize
        norm_c = tf.image.per_image_standardization(img_c_resized).numpy()
        norm_n = tf.image.per_image_standardization(img_n_resized).numpy()

        images_c_3d.append(norm_c)
        images_n_3d.append(norm_n)

    if not images_c_3d:
        return (
            np.empty((0, img_height, img_width, 3), dtype=np.float32),
            np.empty((0, img_height, img_width, 3), dtype=np.float32),
            [],
            [],
        )

    return (
        np.array(images_n_3d, dtype=np.float32),
        np.array(images_c_3d, dtype=np.float32),
        stats_n,
        stats_c,
    )


def slices_to_magnitude(
    images_3d: np.ndarray,
    stats: Optional[List[Tuple[float, float]]] = None,
    orig_f_bins: int = 129,
    reps_horizontal: int = 8,
) -> np.ndarray:
    """
    Inverts a sequence of 3D spectrogram images back into a continuous 2D magnitude STFT matrix.
    Images are shaped (N, 256, 256, 3).
    """
    reconstructed_slices = []

    for idx, img in enumerate(images_3d):
        # Denormalize if standardization statistics are provided
        if stats is not None and idx < len(stats):
            mean, std = stats[idx]
            img = img * max(std, 1e-6) + mean

        # Resize back from (256, 256, 3) to (258, 256, 3)
        img_expanded = tf.image.resize(
            img, [orig_f_bins * 2, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        ).numpy()

        # Each channel corresponds to one slice: c=0 (t), c=1 (t+1), c=2 (t+2)
        for c in range(3):
            slice_v = img_expanded[:, :, c]
            # Top half corresponds to the original frequency bins (0..129)
            slice_h = slice_v[:orig_f_bins, :]
            # Subsample along horizontal time dimension
            slice_orig = slice_h[:, ::reps_horizontal]
            reconstructed_slices.append(np.maximum(slice_orig, 0.0))

    if not reconstructed_slices:
        return np.empty((orig_f_bins, 0), dtype=np.float32)

    return np.concatenate(reconstructed_slices, axis=1)


def reconstruct_waveform(
    magnitude: np.ndarray,
    phase: Optional[np.ndarray] = None,
    sr: int = 16000,
    nfft: int = 256,
    noverlap: int = 192,
    window: str = "hann",
    n_iter: int = 32,
) -> np.ndarray:
    """
    Reconstructs time-domain audio from magnitude STFT.
    If phase is provided (e.g. from bone STFT phase transfer), uses istft.
    Otherwise uses Griffin-Lim.
    """
    win = signal.windows.hann(nfft) if window == "hann" else signal.get_window(window, nfft)

    if phase is not None:
        # Match time length
        min_time = min(magnitude.shape[1], phase.shape[1])
        mag_matched = magnitude[:, :min_time]
        phase_matched = phase[:, :min_time]
        Zxx = mag_matched * np.exp(1j * phase_matched)
        _, audio = signal.istft(Zxx, sr, window=win, nperseg=nfft, noverlap=noverlap)
    else:
        # Griffin-Lim reconstruction
        audio = librosa.griffinlim(
            magnitude,
            n_iter=n_iter,
            hop_length=nfft - noverlap,
            win_length=nfft,
            window=window,
        )

    return audio.astype(np.float32)


def get_paired_audio_files(clean_folder: str, noise_folder: str) -> List[Tuple[str, str]]:
    """
    Finds and pairs clean and noisy/bone audio files.
    Matches identical filenames first; falls back to natural-sorted 1:1 matching.
    """
    if not os.path.exists(clean_folder):
        raise FileNotFoundError(f"Clean audio folder not found: {clean_folder}")
    if not os.path.exists(noise_folder):
        raise FileNotFoundError(f"Noisy/bone audio folder not found: {noise_folder}")

    clean_files = [
        f for f in os.listdir(clean_folder) if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]
    noise_files = [
        f for f in os.listdir(noise_folder) if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]

    clean_files = natsorted(clean_files)
    noise_files = natsorted(noise_files)

    # Check for direct filename intersection
    common_names = set(clean_files).intersection(set(noise_files))
    if len(common_names) > 0 and len(common_names) == len(clean_files) == len(noise_files):
        sorted_common = natsorted(list(common_names))
        return [(name, name) for name in sorted_common]

    # Fallback: pair by sorted order
    pairs = []
    min_count = min(len(clean_files), len(noise_files))
    if len(clean_files) != len(noise_files):
        print(
            f"[Warning] Mismatched file counts: clean={len(clean_files)}, noisy={len(noise_files)}. Pairing first {min_count} files."
        )

    for i in range(min_count):
        pairs.append((clean_files[i], noise_files[i]))

    return pairs


def extract_speaker_id(filename: str) -> str:
    """
    Attempts to extract speaker ID from filename (e.g. 'p225_001.wav' -> 'p225',
    'speaker1_clip02.wav' -> 'speaker1'). If no prefix is detectable, treats
    the stem as unique.
    """
    stem = os.path.splitext(filename)[0]
    match = re.match(r"^([A-Za-z0-9]+)[-_]", stem)
    if match:
        return match.group(1)
    return stem


def create_splits(
    clean_folder: str,
    noise_folder: str,
    splits_dir: str = "splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_by: str = "recording",
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Splits paired audio files into train, val, and test partitions by recording
    (or speaker), guaranteeing zero frame leakage between sets.
    Saves file lists to splits/train.txt, splits/val.txt, and splits/test.txt.
    """
    pairs = get_paired_audio_files(clean_folder, noise_folder)
    if not pairs:
        raise ValueError(f"No audio file pairs found in {clean_folder} and {noise_folder}")

    # Group recordings
    rng = random.Random(seed)
    if split_by == "speaker":
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for c_file, n_file in pairs:
            spk = extract_speaker_id(c_file)
            groups.setdefault(spk, []).append((c_file, n_file))
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio)) if n_groups >= 3 else 0

        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train : n_train + n_val]
        test_keys = group_keys[n_train + n_val :]

        train_pairs = [p for k in train_keys for p in groups[k]]
        val_pairs = [p for k in val_keys for p in groups[k]]
        test_pairs = [p for k in test_keys for p in groups[k]]
    else:
        # Split by recording
        shuffled_pairs = list(pairs)
        rng.shuffle(shuffled_pairs)

        n_total = len(shuffled_pairs)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0

        train_pairs = shuffled_pairs[:n_train]
        val_pairs = shuffled_pairs[n_train : n_train + n_val]
        test_pairs = shuffled_pairs[n_train + n_val :]

    # If test_pairs is empty due to small dataset size, ensure at least 1 in val and test
    if len(test_pairs) == 0 and len(train_pairs) > 2:
        test_pairs.append(train_pairs.pop())
    if len(val_pairs) == 0 and len(train_pairs) > 2:
        val_pairs.append(train_pairs.pop())

    os.makedirs(splits_dir, exist_ok=True)
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split_name, split_list in splits.items():
        split_path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(split_path, "w", encoding="utf-8") as f:
            for c_name, n_name in split_list:
                f.write(f"{c_name}\t{n_name}\n")
        print(f"Saved {len(split_list)} recordings to {split_path}")

    return splits


def load_split_file_list(splits_dir: str, split_name: str) -> List[Tuple[str, str]]:
    """
    Reads a split file (train.txt, val.txt, test.txt).
    Each line contains tab-separated clean_filename and noise_filename,
    or just filename if both are identical.
    """
    split_path = os.path.join(splits_dir, f"{split_name}.txt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                pairs.append((parts[0].strip(), parts[0].strip()))
    return pairs


def create_dataset_from_files(
    clean_folder: str,
    noise_folder: str,
    file_list: List[Tuple[str, str]],
    batch_size: int = 1,
    buffer_size: int = 400,
    is_training: bool = True,
    seed: int = 42,
    stft_config: Optional[dict] = None,
) -> tf.data.Dataset:
    """
    Processes audio recordings in file_list into paired 3D spectrogram image tensors
    and constructs a tf.data.Dataset.
    """
    if stft_config is None:
        stft_config = {
            "sr": 16000,
            "nfft": 256,
            "noverlap": 192,
            "window": "hann",
            "reps_horizontal": 8,
            "reps_vertical": 2,
            "slice_time_frames": 32,
            "img_height": 256,
            "img_width": 256,
        }

    all_inputs = []
    all_targets = []

    for clean_file, noise_file in file_list:
        clean_path = os.path.join(clean_folder, clean_file)
        noise_path = os.path.join(noise_folder, noise_file)

        clean_audio, _ = load_audio(clean_path, sr=stft_config.get("sr", 16000))
        noise_audio, _ = load_audio(noise_path, sr=stft_config.get("sr", 16000))

        slices_c, slices_n, _, _ = extract_slices(
            clean_audio=clean_audio,
            noise_audio=noise_audio,
            sr=stft_config.get("sr", 16000),
            nfft=stft_config.get("nfft", 256),
            noverlap=stft_config.get("noverlap", 192),
            window=stft_config.get("window", "hann"),
            reps_horizontal=stft_config.get("reps_horizontal", 8),
            reps_vertical=stft_config.get("reps_vertical", 2),
            slice_time_frames=stft_config.get("slice_time_frames", 32),
        )

        if len(slices_c) >= 3:
            inputs, targets, _, _ = prepare_3d_images(
                slices_c=slices_c,
                slices_n=slices_n,
                img_height=stft_config.get("img_height", 256),
                img_width=stft_config.get("img_width", 256),
            )
            if len(inputs) > 0:
                all_inputs.append(inputs)
                all_targets.append(targets)

    if not all_inputs:
        # Empty dataset fallback
        empty_x = tf.zeros((0, 256, 256, 3), dtype=tf.float32)
        empty_y = tf.zeros((0, 256, 256, 3), dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices((empty_x, empty_y)).batch(batch_size)

    inputs_tensor = np.concatenate(all_inputs, axis=0)
    targets_tensor = np.concatenate(all_targets, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((inputs_tensor, targets_tensor))

    if is_training:
        dataset = dataset.shuffle(buffer_size, seed=seed)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create splits or verify data pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--create-splits", action="store_true", help="Generate train/val/test splits")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    if args.create_splits:
        d_cfg = cfg.get("data", {})
        create_splits(
            clean_folder=d_cfg.get("clean_folder", "data/clean"),
            noise_folder=d_cfg.get("noise_folder", "data/noisy"),
            splits_dir=d_cfg.get("splits_dir", "splits"),
            train_ratio=d_cfg.get("train_ratio", 0.70),
            val_ratio=d_cfg.get("val_ratio", 0.15),
            test_ratio=d_cfg.get("test_ratio", 0.15),
            seed=cfg.get("seed", 42),
            split_by=d_cfg.get("split_by", "recording"),
        )
