"""
Evaluation script for BoneGAN.
Evaluates on the test split only, computing PESQ, STOI, and LSD (Log-Spectral Distance).
Writes metrics with mean ± std across test clips to results/metrics.json.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy import signal
import soundfile as sf
import tensorflow as tf

from data import (
    extract_slices,
    load_audio,
    load_split_file_list,
    prepare_3d_images,
    reconstruct_waveform,
    slices_to_magnitude,
)
from model import Generator
from utils import load_config, set_seed


def compute_lsd(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    sr: int = 16000,
    nfft: int = 256,
    noverlap: int = 192,
) -> float:
    """
    Computes Log-Spectral Distance (LSD) in dB between reference and degraded/predicted audio:
    LSD = (1/T) * sum_t sqrt( (1/K) * sum_k [ 20 * log10( |S(k,t)| / |S_hat(k,t)| ) ]^2 )
    """
    win = signal.windows.hann(nfft)
    _, _, Z_ref = signal.stft(ref_audio, sr, window=win, nperseg=nfft, noverlap=noverlap)
    _, _, Z_deg = signal.stft(deg_audio, sr, window=win, nperseg=nfft, noverlap=noverlap)

    mag_ref = np.abs(Z_ref)
    mag_deg = np.abs(Z_deg)

    min_time = min(mag_ref.shape[1], mag_deg.shape[1])
    mag_ref = mag_ref[:, :min_time]
    mag_deg = mag_deg[:, :min_time]

    eps = 1e-12
    log_ratio = 20.0 * np.log10((mag_ref + eps) / (mag_deg + eps))
    frame_lsd = np.sqrt(np.mean(log_ratio**2, axis=0))
    return float(np.mean(frame_lsd))


def evaluate(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    save_audio: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Runs model evaluation strictly on the test split.
    """
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg.get("data", {})
    stft_cfg = cfg.get("stft", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})

    clean_folder = data_cfg.get("clean_folder", "data/clean")
    noise_folder = data_cfg.get("noise_folder", "data/noisy")
    splits_dir = data_cfg.get("splits_dir", "splits")

    results_dir = eval_cfg.get("results_dir", "results")
    if output_dir:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = eval_cfg.get("metrics_file", os.path.join(results_dir, "metrics.json"))

    # Load test split
    test_files = load_split_file_list(splits_dir, "test")
    if not test_files:
        raise ValueError(f"No test files found in {splits_dir}/test.txt")

    print(f"Loaded {len(test_files)} recordings from test split.")

    # Initialize generator
    img_height = model_cfg.get("img_height", 256)
    img_width = model_cfg.get("img_width", 256)
    in_channels = model_cfg.get("input_channels", 3)
    out_channels = model_cfg.get("output_channels", 3)

    generator = Generator(
        input_shape=(img_height, img_width, in_channels),
        output_channels=out_channels,
    )

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(generator=generator)
    if os.path.isdir(checkpoint_path):
        resolved_ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if not resolved_ckpt:
            raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_path}")
    else:
        resolved_ckpt = checkpoint_path

    print(f"Restoring generator from: {resolved_ckpt}")
    status = ckpt.restore(resolved_ckpt)
    status.expect_partial()

    sr = stft_cfg.get("sr", 16000)
    nfft = stft_cfg.get("nfft", 256)
    noverlap = stft_cfg.get("noverlap", 192)
    window = stft_cfg.get("window", "hann")
    reps_h = stft_cfg.get("reps_horizontal", 8)
    reps_v = stft_cfg.get("reps_vertical", 2)
    slice_frames = stft_cfg.get("slice_time_frames", 32)

    pesq_scores = []
    stoi_scores = []
    lsd_scores = []
    clip_details = []

    for idx, (clean_file, noise_file) in enumerate(test_files):
        clean_path = os.path.join(clean_folder, clean_file)
        noise_path = os.path.join(noise_folder, noise_file)

        clean_audio, _ = load_audio(clean_path, sr=sr)
        noise_audio, _ = load_audio(noise_path, sr=sr)

        slices_c, slices_n, phase_noisy, _ = extract_slices(
            clean_audio=clean_audio,
            noise_audio=noise_audio,
            sr=sr,
            nfft=nfft,
            noverlap=noverlap,
            window=window,
            reps_horizontal=reps_h,
            reps_vertical=reps_v,
            slice_time_frames=slice_frames,
        )

        if len(slices_n) < 3:
            print(f"[Warning] Skipping {clean_file}: insufficient frames ({len(slices_n)} slices).")
            continue

        inputs, targets, stats_n, stats_c = prepare_3d_images(
            slices_c=slices_c,
            slices_n=slices_n,
            img_height=img_height,
            img_width=img_width,
        )

        # Inference
        gen_outputs = generator(inputs, training=False).numpy()

        # Reconstruct magnitude and waveform
        gen_mag = slices_to_magnitude(
            gen_outputs,
            stats=stats_c,
            orig_f_bins=129,
            reps_horizontal=reps_h,
        )
        pred_audio = reconstruct_waveform(
            gen_mag,
            phase=phase_noisy,
            sr=sr,
            nfft=nfft,
            noverlap=noverlap,
            window=window,
        )

        min_len = min(len(clean_audio), len(pred_audio))
        ref_cut = clean_audio[:min_len]
        pred_cut = pred_audio[:min_len]

        # Compute PESQ
        try:
            p_val = float(pesq(sr, ref_cut, pred_cut, "wb"))
        except Exception:
            try:
                p_val = float(pesq(sr, ref_cut, pred_cut, "nb"))
            except Exception as e:
                print(f"[Warning] PESQ failed on {clean_file}: {e}")
                p_val = float("nan")

        # Compute STOI
        try:
            s_val = float(stoi(ref_cut, pred_cut, sr, extended=False))
        except Exception as e:
            print(f"[Warning] STOI failed on {clean_file}: {e}")
            s_val = float("nan")

        # Compute LSD
        try:
            l_val = compute_lsd(ref_cut, pred_cut, sr=sr, nfft=nfft, noverlap=noverlap)
        except Exception as e:
            print(f"[Warning] LSD failed on {clean_file}: {e}")
            l_val = float("nan")

        if not np.isnan(p_val):
            pesq_scores.append(p_val)
        if not np.isnan(s_val):
            stoi_scores.append(s_val)
        if not np.isnan(l_val):
            lsd_scores.append(l_val)

        clip_info = {
            "clean_file": clean_file,
            "noise_file": noise_file,
            "pesq": p_val,
            "stoi": s_val,
            "lsd": l_val,
        }
        clip_details.append(clip_info)

        if save_audio:
            save_path = os.path.join(results_dir, f"enhanced_{os.path.basename(clean_file)}")
            sf.write(save_path, pred_cut, sr)

    def stats_dict(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    metrics = {
        "pesq": stats_dict(pesq_scores),
        "stoi": stats_dict(stoi_scores),
        "lsd": stats_dict(lsd_scores),
        "num_clips": len(clip_details),
        "clips": clip_details,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n================ Test Set Evaluation Results ================")
    print(f"Total Test Clips Evaluated: {metrics['num_clips']}")
    print(f"PESQ (WB): {metrics['pesq']['mean']:.4f} ± {metrics['pesq']['std']:.4f}")
    print(f"STOI:      {metrics['stoi']['mean']:.4f} ± {metrics['stoi']['std']:.4f}")
    print(f"LSD (dB):  {metrics['lsd']['mean']:.4f} ± {metrics['lsd']['std']:.4f}")
    print(f"Metrics saved to: {metrics_file}")
    print("=============================================================\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BoneGAN on test split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory or prefix (e.g. training_checkpoints or ckpt-10)",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--save-audio", action="store_true", help="Save reconstructed enhanced wav files"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_audio=args.save_audio,
        output_dir=args.output_dir,
    )
