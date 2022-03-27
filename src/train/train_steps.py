from typing import Any
import os
import functools
import jax
import flax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from src.models import metrics
from flax import jax_utils
import gc
import tensorflow as tf
import wandb
from absl import app
from absl import flags

@functools.partial(jax.jit, static_argnames=("prefix", "metrics_to_calc"))
def train_step_bn_dsc_loss(state, batch_images, batch_labels, prefix, metrics_to_calc):
  def dice_coefficient_loss(params, smooth=.0000001):
    prediction, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                      batch_images,
                                                      mutable=['batch_stats'],
                                                      is_training=True)
    y_true_flat = jnp.ravel(batch_labels)
    y_pred_flat = jnp.ravel(prediction)
    intersection = jnp.sum(y_true_flat * y_pred_flat)
    return -(2. * intersection + smooth) / (jnp.sum(y_true_flat) + jnp.sum(y_pred_flat) + smooth), (new_model_state, prediction)
  grad_function = jax.value_and_grad(dice_coefficient_loss, has_aux=True)
  (train_loss, (new_model_state, logits)), grads = grad_function(state.params)
  batch_metrics = metrics.calculate_metrics(batch_labels,
                                                    jnp.squeeze(logits, axis=-1),
                                                    metrics_to_calc,
                                                    prefix=prefix)
  return grads, batch_metrics, new_model_state


@functools.partial(jax.jit, static_argnames=("prefix", "metrics_to_calc"))
def train_step_bn_multi_scale_dsc_loss(state, batch_images, batch_labels, prefix, metrics_to_calc):
  def dice_coefficient_loss(params, smooth=.0000001):
    predictions, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                      jnp.squeeze(batch_images),
                                                      mutable=['batch_stats'],
                                                      is_training=True)
    dices = []
    for prediction, labels in zip(predictions, batch_labels):
      y_true_flat = jnp.ravel(labels)
      y_pred_flat = jnp.ravel(prediction)
      intersection = jnp.sum(y_true_flat * y_pred_flat)
      dices.append(-(2. * intersection + smooth) / (jnp.sum(y_true_flat) + jnp.sum(y_pred_flat) + smooth))
    return jnp.mean(jnp.array(dices)), (new_model_state, predictions[0])
  grad_function = jax.value_and_grad(dice_coefficient_loss, has_aux=True)
  (train_loss, (new_model_state, logits)), grads = grad_function(state.params)
  batch_metrics = metrics.calculate_metrics(batch_labels[0],
                                            jnp.squeeze(logits, axis=-1),
                                            metrics_to_calc,
                                            prefix=prefix)
  return grads, batch_metrics, new_model_state