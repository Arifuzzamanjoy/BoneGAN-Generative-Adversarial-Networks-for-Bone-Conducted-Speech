"""
Utility functions for reproducibility, configuration loading, and logging.
"""

import os
import random
import yaml
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """
    Sets seeds across Python random, NumPy, and TensorFlow for full reproducibility,
    and enables TensorFlow deterministic operations.

    Note on deterministic operations:
    `tf.config.experimental.enable_op_determinism()` enforces deterministic GPU/CPU
    kernels. Most TensorFlow operations support determinism. On GPU with CUDA, certain
    backward passes or atomic operations (e.g., custom CUDA reductions or specific
    historical Conv2D/Conv2DTranspose implementations with certain cuDNN algorithms)
    may not support deterministic execution or may raise an UnimplementedError if no
    deterministic path exists. On CPU and modern TF (>=2.15), standard operations
    execute deterministically once enabled.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"[Warning] Could not enable TensorFlow op determinism: {e}")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration settings from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_path: str) -> None:
    """
    Saves a configuration dictionary to a YAML file next to a run or checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

