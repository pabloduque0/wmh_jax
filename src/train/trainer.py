from typing import Any
import os

#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2/"
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.2/:$LD_LIBRARY_PATH"

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/x86_64-linux-gnu/"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda/targets/x86_64-linux/lib/"
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
#os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"]= "cuda_malloc_async"

import functools
from src import utils
import flax.linen as nn
import src.constants as cons
import joblib
import optax
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
import pandas as pd
import inspect

from src.models import old_unet
from src.models import multi_scale_unet
from src.models import swin_unet

from src.configs import configs_old_unet
from src.configs import configs_multi_scale_unet
from src.configs import configs_swin_unet

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

MODELS_REL_PATH = "models/"
FLAGS = flags.FLAGS

flags.DEFINE_integer('fold_index', None, 'Fold index')

class TrainState(train_state.TrainState):
  batch_stats: Any = None

def prepare_tf_data(xs):
  # From FLAX examples: https://github.com/google/flax/blob/46126954dce1beee5b200586e983af5794137ca0/examples/imagenet/train.py#L175
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)

def prepare_train_dataset(dataset, epochs, batch_size):
  dataset = dataset.repeat(epochs)
  dataset = dataset.shuffle(buffer_size=batch_size)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  iterator_ = map(prepare_tf_data, dataset)
  iterator_ = jax_utils.prefetch_to_device(iterator=iterator_,
                                    size=1,
                                    devices=[jax.devices()[0]])
  return iterator_

def prepare_val_dataset(dataset, epochs, batch_size):
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.repeat(epochs)
  iterator_ = map(prepare_tf_data, dataset)
  iterator_ = jax_utils.prefetch_to_device(iterator=iterator_,
                                    size=1,
                                    devices=[jax.devices()[0]])
  return iterator_


def train_flax(
    train_dataset,
    validation_dataset,
    state,
    metrics_to_calc,
    steps_per_epoch,
    steps_per_eval,
    aug_function,
    train_step_func,
    eval_step_func,
    wandb_run):
  
  print("Steps: ", steps_per_epoch, steps_per_eval)
  train_metrics = []
  validation_metrics = []
  for step, (batch_images, batch_labels) in enumerate(train_dataset):
    epoch = step // steps_per_epoch
    aug_images, aug_labels = aug_function(images=batch_images,
                                          labels=batch_labels)
    state, train_batch_metrics = train_step_func(state=state,
                                       batch_images=aug_images,#jnp.squeeze(batch_images, axis=0),
                                       batch_labels=aug_labels,#jnp.squeeze(batch_labels, axis=0),
                                       metrics_to_calc=metrics_to_calc,
                                       prefix="train")
    #print(jax.tree_map(lambda x: np.isnan(x).any(), grads))
    #wandb_run.log(grads.unfreeze(), step=epoch, commit=False)
    train_metrics.append(train_batch_metrics)
    if (step + 1) % steps_per_epoch == 0:
      print("Epoch: ", epoch)
      for i, (val_batch_images, val_batch_labels) in zip(range(steps_per_eval), validation_dataset):
        validation_batch_metrics = eval_step_func(state=state,
                                                        batch_images=jnp.squeeze(val_batch_images, axis=0),
                                                        batch_labels=val_batch_labels,
                                                        metrics_to_calc=metrics_to_calc,
                                                        prefix="validation")
        validation_metrics.append(validation_batch_metrics)

      train_epoch_metrics = jax.tree_map(lambda x: jnp.mean(x), common_utils.stack_forest(train_metrics))
      validation_epoch_metrics = jax.tree_map(lambda x: jnp.mean(x), common_utils.stack_forest(validation_metrics))
      wandb_run.log(train_epoch_metrics, step=epoch, commit=False)
      wandb_run.log(validation_epoch_metrics, step=epoch)
      print("Train metrics: ", train_epoch_metrics)
      print("Validation metrics: ", validation_epoch_metrics)
  del train_metrics, validation_metrics


def main(argv):
    i = FLAGS.fold_index
    cpu = jax.devices('cpu')[0]
    
    config = configs_swin_unet.get_augment_config_2()
    model = swin_unet.SwinTransformerSys(**config.model_config)

    metrics_to_calc = (metrics.dice_coefficient,
                       metrics.average_volume_difference,)
                       #metrics.lession_recall,
                       #metrics.lession_precision,
                       #metrics.lession_f1)
    config_for_log = config.to_dict()
    config_for_log["train_config"]["eval_step_func"] = config_for_log["train_config"]["eval_step_func"].__wrapped__
    config_for_log["train_config"]["train_step_func"] = config_for_log["train_config"]["train_step_func"].__wrapped__
    wandb_run = wandb.init(project="wmh", entity="pabloduque0", reinit=True, config=config_for_log)
    if config.general_config.multi_label:
        train_labels_name = cons.MULTI_LABELS_TRAIN_NAME
        validation_labels_name = cons.MULTI_LABELS_VALIDATION_NAME
    else:
        train_labels_name = cons.LABELS_TRAIN_NAME
        validation_labels_name = cons.LABELS_VALIDATION_NAME
    
    print("LOADING DATA...")

    train_data, train_labels = utils.load_preprocessed_data(config.general_config.data_path,
                                                            cons.TRAIN_FOLDER,
                                                            cons.DATA_TRAIN_NAME,
                                                            train_labels_name,
                                                            i)

    validation_data, validation_labels = utils.load_preprocessed_data(config.general_config.data_path,
                                                                      cons.VALIDATION_FOLDER,
                                                                      cons.DATA_VALIDATION_NAME,
                                                                      validation_labels_name,
                                                                      i)
    gc.collect()

    single_training_name = '{}_{}'.format(config.group_name, i)
    #utils.create_folders(single_training_name, MODELS_REL_PATH)
    if config.general_config.pad_crop_function is not None:
      train_data = config.general_config.pad_crop_function(train_data, **config.general_config.pad_crop_kwargs)
      train_labels = config.general_config.pad_crop_function(train_labels, **config.general_config.pad_crop_kwargs)
      validation_data = config.general_config.pad_crop_function(validation_data, **config.general_config.pad_crop_kwargs)
      validation_labels = config.general_config.pad_crop_function(validation_labels, **config.general_config.pad_crop_kwargs)

    print(train_data.shape, validation_data.shape)
    steps_per_epoch = train_data.shape[0] // config.train_config.batch_size
    steps_per_eval = validation_data.shape[0] // config.train_config.batch_size
    if config.general_config.multi_label:
      train_labels = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(sub_data) for sub_data in train_labels]))
      validation_labels = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(sub_data) for sub_data in validation_labels]))
    else:
      train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
      validation_labels = tf.data.Dataset.from_tensor_slices(validation_labels)

    train_data = tf.data.Dataset.from_tensor_slices(train_data)#[:, 4:196, 4:196, :]
    validation_data = tf.data.Dataset.from_tensor_slices(validation_data)
    train_dataset = prepare_train_dataset(tf.data.Dataset.zip((train_data, train_labels)),
                                    epochs=config.train_config.epochs,
                                    batch_size=config.train_config.batch_size)
    validation_dataset = prepare_val_dataset(tf.data.Dataset.zip((validation_data, validation_labels)),
                                          epochs=config.train_config.epochs,
                                          batch_size=config.train_config.batch_size)
    del train_data, train_labels, validation_data, validation_labels
    gc.collect()
    print("INIT MODEL...")
    rng = jax.random.PRNGKey(0)
    print("USING DEVICE: ", jax.default_backend(), jax.devices())
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, **config.general_config.call_kwargs)
    state = TrainState.create(apply_fn=model.apply,
                                          params=variables["params"],
                                          tx=config.train_config.optimizer(**config.optimizer_config),
                                          **{"batch_stats": variables["batch_stats"]} if "batch_stats" in variables else {})
    print("MODELS PARAMS: ", sum(x.size for x in jax.tree_leaves(variables["params"])))
    train_flax(train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                state=state,
                metrics_to_calc=metrics_to_calc,
                steps_per_epoch=steps_per_epoch,
                steps_per_eval=steps_per_eval,
                aug_function=functools.partial(config.augment_config.aug_function, m=config.augment_config.m, n=config.augment_config.n),
                train_step_func=config.train_config.train_step_func,
                eval_step_func=config.train_config.eval_step_func,
                wandb_run=wandb_run)

    #full_model_path = os.path.join(MODELS_REL_PATH, "models", single_training_name)
    #url_path = os.path.join(full_model_path, cons.EXPERIMENT_ULR_FILENAME)
    #joblib.dump(wandb_run.url, url_path)
    wandb_run.finish()

    del train_dataset, validation_dataset, model, variables, state
    gc.collect()



if __name__ == '__main__':
  flags.mark_flag_as_required('fold_index')
  app.run(main)