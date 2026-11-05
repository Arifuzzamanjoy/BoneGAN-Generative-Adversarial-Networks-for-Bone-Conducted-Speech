"""
Loss functions for Pix2Pix BoneGAN training.
Preserves the exact loss mathematics from the original formulation:
- generator_loss = adversarial BCE + LAMBDA * L1
- discriminator_loss = real BCE + generated BCE
"""

from typing import Tuple
import tensorflow as tf

# Standard Binary Cross Entropy loss from logits
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(
    disc_generated_output: tf.Tensor,
    gen_output: tf.Tensor,
    target: tf.Tensor,
    lambda_l1: float = 100.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes total generator loss as sum of adversarial BCE and weighted L1 loss:
        total_gen_loss = gan_loss + lambda_l1 * l1_loss

    Returns:
        (total_gen_loss, gan_loss, l1_loss)
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(
    disc_real_output: tf.Tensor,
    disc_generated_output: tf.Tensor,
) -> tf.Tensor:
    """
    Computes discriminator loss as sum of BCE on real pairs and BCE on generated pairs:
        total_disc_loss = real_loss + generated_loss

    Returns:
        total_disc_loss
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

