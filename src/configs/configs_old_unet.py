import ml_collections
import flax.linen as nn
from src.models import metrics
from src.augmentation import augmentation_functions
import optax
from src.train import eval_steps
from src.train import train_steps
import jax.numpy as jnp
from jax import random
from src import utils



def get_initial_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "OldUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.pad_crop_function = None
  config.general_config.pad_crop_kwargs = None
  config.general_config.call_kwargs = {"x": jnp.ones((1, 200, 200, 2)), "is_training": True}
  config.general_config.key_rngs = {}

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.train_step_func = train_steps.train_step_bn_dsc_loss
  config.train_config.eval_step_func = eval_steps.eval_step_bn
  config.train_config.optimizer = optax.adam
  config.train_config.batch_size = 10
  config.train_config.epochs = 100
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()

  config.model_config.stack_features = 80
  config.model_config.stack_kernel_size = (7, 7)
  config.model_config.stack_n_convolutions = 3
  config.model_config.down_features = [128, 140, 160]
  config.model_config.down_kernel_size = [(7, 7), (7, 7) ]
  config.model_config.down_window_shape = [(2, 2), (2, 2)]
  config.model_config.n_steps = 2
  config.model_config.bottom_features = [200, 200]
  config.model_config.bottom_kernel_size = [(7, 7), (7, 7)]
  config.model_config.n_bottom_convs = 2
  config.model_config.up_features = [160, 140, 128]
  config.model_config.up_kernel_size = [(7, 7), (7, 7)]
  config.model_config.attn_gating_features = [100, 100, 100]
  config.model_config.attn_gating_kernel_size = [(7, 7), (7, 7)]
  config.model_config.last_stack_features = 80
  config.model_config.last_stack_kernel_size = (7, 7)
  config.model_config.last_stack_n_convolutions = 2

  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.000005

  config.augment_config = ml_collections.ConfigDict()
  config.augment_config.aug_function = augmentation_functions.no_augmentation
  config.augment_config.m = None
  config.augment_config.n = None

  hash_ = "test"#utils.dict_hash(config.to_dict())
  config.group_name = f"base_old_unet_{hash_}"

  return config


def get_initial_2_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.group_name = "base_old_unet"
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "OldUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.levels_multilabel = 3

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.optimizer = optax.adamw
  config.train_config.batch_size = 10
  config.train_config.epochs = 45
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()

  config.model_config.stack_features = 80
  config.model_config.stack_kernel_size = (7, 7)
  config.model_config.stack_n_convolutions = 3
  config.model_config.down_features = [128, 140, 160]
  config.model_config.down_kernel_size = [(7, 7), (7, 7) ]
  config.model_config.down_window_shape = [(2, 2), (2, 2)]
  config.model_config.n_steps = 2
  config.model_config.bottom_features = [200, 200]
  config.model_config.bottom_kernel_size = [(7, 7), (7, 7)]
  config.model_config.n_bottom_convs = 2
  config.model_config.up_features = [160, 140, 128]
  config.model_config.up_kernel_size = [(7, 7), (7, 7)]
  config.model_config.attn_gating_features = [100, 100, 100]
  config.model_config.attn_gating_kernel_size = [(7, 7), (7, 7)]
  config.model_config.last_stack_features = 80
  config.model_config.last_stack_kernel_size = (7, 7)
  config.model_config.last_stack_n_convolutions = 2

  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.000005
  config.optimizer_config.weight_decay = 0.0000001

  return config