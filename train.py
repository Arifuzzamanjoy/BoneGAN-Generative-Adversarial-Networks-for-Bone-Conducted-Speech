"""
Training script for BoneGAN.
Handles GradientTape training loop, checkpoints, and per-epoch logging.
"""

import argparse
import os
import time
from typing import Optional

import tensorflow as tf

from data import (
    create_dataset_from_files,
    create_splits,
    load_split_file_list,
)
from losses import discriminator_loss, generator_loss
from model import Discriminator, Generator
from utils import load_config, save_config, set_seed


def train(config_path: str = "config.yaml", **overrides) -> str:
    """
    Main training function for BoneGAN.
    """
    cfg = load_config(config_path)

    # Apply any command-line overrides
    if overrides.get("epochs") is not None:
        cfg["training"]["epochs"] = overrides["epochs"]
    if overrides.get("batch_size") is not None:
        cfg["training"]["batch_size"] = overrides["batch_size"]
    if overrides.get("checkpoint_dir") is not None:
        cfg["training"]["checkpoint_dir"] = overrides["checkpoint_dir"]
    if overrides.get("seed") is not None:
        cfg["seed"] = overrides["seed"]

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Unpack configurations
    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    opt_cfg = cfg.get("optimizer", {})
    stft_cfg = cfg.get("stft", {})
    model_cfg = cfg.get("model", {})

    epochs = training_cfg.get("epochs", 100)
    batch_size = training_cfg.get("batch_size", 1)
    buffer_size = training_cfg.get("buffer_size", 400)
    checkpoint_dir = training_cfg.get("checkpoint_dir", "training_checkpoints")
    checkpoint_prefix = training_cfg.get("checkpoint_prefix", "ckpt")
    log_dir = training_cfg.get("log_dir", "logs")
    save_every_epochs = training_cfg.get("save_every_epochs", 5)

    lambda_l1 = float(loss_cfg.get("lambda_l1", 100.0))
    gen_lr = float(opt_cfg.get("generator_lr", 2e-4))
    disc_lr = float(opt_cfg.get("discriminator_lr", 2e-4))
    beta_1 = float(opt_cfg.get("beta_1", 0.5))
    beta_2 = float(opt_cfg.get("beta_2", 0.999))

    clean_folder = data_cfg.get("clean_folder", "data/clean")
    noise_folder = data_cfg.get("noise_folder", "data/noisy")
    splits_dir = data_cfg.get("splits_dir", "splits")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Requirement 4: Log config next to every run
    run_config_path = os.path.join(checkpoint_dir, "config.yaml")
    save_config(cfg, run_config_path)
    print(f"Logged configuration to {run_config_path}")

    # Ensure splits exist
    train_split_file = os.path.join(splits_dir, "train.txt")
    val_split_file = os.path.join(splits_dir, "val.txt")
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print(f"Splits not found in {splits_dir}. Creating leakage-safe splits...")
        create_splits(
            clean_folder=clean_folder,
            noise_folder=noise_folder,
            splits_dir=splits_dir,
            train_ratio=data_cfg.get("train_ratio", 0.70),
            val_ratio=data_cfg.get("val_ratio", 0.15),
            test_ratio=data_cfg.get("test_ratio", 0.15),
            seed=seed,
            split_by=data_cfg.get("split_by", "recording"),
        )

    train_files = load_split_file_list(splits_dir, "train")
    val_files = load_split_file_list(splits_dir, "val") if os.path.exists(val_split_file) else []

    print(f"Loaded {len(train_files)} training recordings, {len(val_files)} validation recordings.")

    train_ds = create_dataset_from_files(
        clean_folder=clean_folder,
        noise_folder=noise_folder,
        file_list=train_files,
        batch_size=batch_size,
        buffer_size=buffer_size,
        is_training=True,
        seed=seed,
        stft_config=stft_cfg,
    )

    val_ds = None
    if val_files:
        val_ds = create_dataset_from_files(
            clean_folder=clean_folder,
            noise_folder=noise_folder,
            file_list=val_files,
            batch_size=batch_size,
            is_training=False,
            seed=seed,
            stft_config=stft_cfg,
        )

    # Initialize models
    img_height = model_cfg.get("img_height", 256)
    img_width = model_cfg.get("img_width", 256)
    in_channels = model_cfg.get("input_channels", 3)
    out_channels = model_cfg.get("output_channels", 3)

    generator = Generator(
        input_shape=(img_height, img_width, in_channels),
        output_channels=out_channels,
    )
    discriminator = Discriminator(
        input_shape=(img_height, img_width, in_channels)
    )

    generator_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=beta_1, beta_2=beta_2)

    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5, checkpoint_name=checkpoint_prefix
    )

    # TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    )

    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target, lambda_l1=lambda_l1
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    print("Starting training...")
    latest_saved_ckpt = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_gen = 0.0
        total_gan = 0.0
        total_l1 = 0.0
        total_disc = 0.0
        steps = 0

        for input_batch, target_batch in train_ds:
            g_total, g_gan, g_l1, d_loss = train_step(input_batch, target_batch)
            total_gen += float(g_total)
            total_gan += float(g_gan)
            total_l1 += float(g_l1)
            total_disc += float(d_loss)
            steps += 1

        if steps == 0:
            print("[Warning] No training batches found in train_ds.")
            break

        avg_gen = total_gen / steps
        avg_gan = total_gan / steps
        avg_l1 = total_l1 / steps
        avg_disc = total_disc / steps
        epoch_time = time.time() - epoch_start

        # Per-epoch logging
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Gen Total: {avg_gen:.4f} | GAN: {avg_gan:.4f} | L1: {avg_l1:.4f} | "
            f"Disc: {avg_disc:.4f} | Time: {epoch_time:.2f}s"
        )

        with summary_writer.as_default():
            tf.summary.scalar("epoch/gen_total_loss", avg_gen, step=epoch)
            tf.summary.scalar("epoch/gen_gan_loss", avg_gan, step=epoch)
            tf.summary.scalar("epoch/gen_l1_loss", avg_l1, step=epoch)
            tf.summary.scalar("epoch/disc_loss", avg_disc, step=epoch)

        if epoch % save_every_epochs == 0 or epoch == epochs:
            latest_saved_ckpt = manager.save()
            print(f"Checkpoint saved: {latest_saved_ckpt}")

    print("Training complete.")
    return latest_saved_ckpt or manager.latest_checkpoint or checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BoneGAN model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Override checkpoint directory"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    train(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
