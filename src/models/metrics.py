from typing import Callable, Mapping

import jax.numpy as jnp
import jax
import numpy as np
from collections import defaultdict
from jax.experimental import jax2tf
import tensorflow as tf
import tensorflow_addons as tfa
import scipy


@jax.jit
def dice_coefficient(y_true, y_pred, smooth: float = .0000001):
  y_true_flat = jnp.ravel(y_true)
  y_pred_flat = jnp.ravel(y_pred)

  intersection = jnp.sum(y_true_flat * y_pred_flat)
  return (2. * intersection + smooth) / (jnp.sum(y_true_flat) + jnp.sum(y_pred_flat) + smooth)

@jax.jit
def dice_loss(y_true, y_pred):
  return -dice_coefficient(y_true=y_true, y_pred=y_pred)

@jax.jit
def average_volume_difference(y_true, y_pred):
  return jnp.abs(y_true.sum() - y_pred.sum()) / y_true.sum() * 100

def connected_components(data):
  data = jnp.squeeze(data)
  conn_comp = jax.experimental.jax2tf.call_tf(tfa.image.connected_components)(data)
  return conn_comp

@jax.jit
def sub_lession_recall(y_true, y_pred):
  pass

def lession_recall(y_true, y_pred):
    conn_comp_true = connected_components(y_true)
    correctly_pred_conn_comps = conn_comp_true * y_pred
    # We add background for the unlikely case there is not one.
    conn_comp_true = jnp.unique(jnp.concatenate([jnp.array([0.]),
                                                   conn_comp_true.ravel()])).shape[0] - 1 # Background does not count
    correctly_pred_conn_comps = jnp.unique(jnp.concatenate([jnp.array([0.]), correctly_pred_conn_comps.ravel()])).shape[0] - 1 # Background does not count
    if conn_comp_true == 0.:
      return 1.
    
    return correctly_pred_conn_comps / conn_comp_true


def lession_precision(y_true, y_pred):
  conn_comp_true = connected_components(y_true)
  correctly_pred_conn_comps = conn_comp_true * y_pred

  total_n_pred = connected_components(y_pred)
  total_n_pred = jnp.unique(jnp.concatenate([jnp.array([0.]), total_n_pred.ravel()])).shape[0] - 1 # Background does not count
  n_correctly_pred_conn_comps = jnp.unique(jnp.concatenate([jnp.array([0.]), correctly_pred_conn_comps.ravel()])).shape[0] - 1 # Background does not count

  if total_n_pred == 0.:
    return 1.
  return n_correctly_pred_conn_comps / total_n_pred


def lession_f1(y_true, y_pred):
  precision = lession_precision(y_true, y_pred)
  recall = lession_recall(y_true, y_pred)
  return 2.0 * (precision * recall) / (precision + recall)


def calculate_metrics(y_true, y_pred, metrics_to_calc, prefix):
  metrics = {}
  for metric in metrics_to_calc:
    metric_value = metric(y_true, y_pred)
    metrics[f"{prefix}/{metric.__name__ }"] = metric_value
  return metrics


