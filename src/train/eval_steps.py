import functools
import jax
import flax
import jax.numpy as jnp
from src.models import metrics


@functools.partial(jax.jit, static_argnames=("prefix", "metrics_to_calc"))
def eval_step_multi_scale_bn(state, batch_images, batch_labels, prefix, metrics_to_calc):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch_images, is_training=False, mutable=False)
  return metrics.calculate_metrics(jnp.squeeze(batch_labels[0]),
                                  jnp.squeeze(logits[0], axis=-1),
                                  metrics_to_calc,
                                  prefix=prefix)

@functools.partial(jax.jit, static_argnames=("prefix", "metrics_to_calc"))
def eval_step_bn(state, batch_images, batch_labels, prefix, metrics_to_calc):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch_images, is_training=False, mutable=False)
  return metrics.calculate_metrics(batch_labels,
                                  jnp.squeeze(logits, axis=-1),
                                  metrics_to_calc,
                                  prefix=prefix)


@functools.partial(jax.jit, static_argnames=("prefix", "metrics_to_calc"))
def eval_step_simple(state, batch_images, batch_labels, prefix, metrics_to_calc):
  logits = state.apply_fn({"params": state.params}, batch_images)
  return metrics.calculate_metrics(jnp.squeeze(batch_labels, axis=0),
                                  jnp.squeeze(logits, axis=-1),
                                  metrics_to_calc,
                                  prefix=prefix)
