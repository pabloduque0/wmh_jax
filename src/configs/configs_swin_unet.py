import ml_collections
import flax.linen as nn
from src.models import metrics
from src.augmentation import augmentation_functions
from src.train import eval_steps
from src.train import train_steps
from src import utils
import jax.numpy as jnp
import optax
from jax import random

def get_initial_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.pad_crop_function = jnp.pad
  config.general_config.pad_crop_kwargs = {"pad_width": ((0, 0), (12, 12), (12, 12), (0, 0)), "mode": "constant", "constant_values": 0}
  config.general_config.call_kwargs = {"x": jnp.ones((1, 224, 224, 2))}
  config.general_config.key_rngs = {"drop_path": random.PRNGKey(0)}

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.train_step_func = train_steps.train_step_dsc_loss
  config.train_config.eval_step_func = eval_steps.eval_step_simple
  config.train_config.optimizer = optax.adam
  config.train_config.batch_size = 8
  config.train_config.epochs = 100
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (224, 224)
  config.model_config.patch_size = (4, 4)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (3, 6, 12, 24)
  config.model_config.window_size = 7
  config.model_config.mlp_ratio = 4.
  config.model_config.qkv_bias = True
  config.model_config.qk_scale = None
  config.model_config.drop_rate = 0.
  config.model_config.attn_drop_rate = 0.
  config.model_config.drop_path_rate = 0.1
  config.model_config.norm_layer = nn.LayerNorm
  config.model_config.ape = False
  config.model_config.patch_norm = True
  config.model_config.use_checkpoint = False
  config.model_config.final_upsample = "expand_first"


  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.0001

  config.augment_config = ml_collections.ConfigDict()
  config.augment_config.aug_function = augmentation_functions.no_augmentation
  config.augment_config.m = None
  config.augment_config.n = None

  hash_ = utils.dict_hash(config.to_dict())
  config.group_name = f"base_swin_unet_{hash_}"

  return config


def get_augment_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.pad_crop_function = jnp.pad
  config.general_config.pad_crop_kwargs = {"pad_width": ((0, 0), (12, 12), (12, 12), (0, 0)), "mode": "constant", "constant_values": 0}
  config.general_config.call_kwargs = {"x": jnp.ones((1, 224, 224, 2))}
  config.general_config.key_rngs = {"drop_path": random.PRNGKey(0)}

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.train_step_func = train_steps.train_step_dsc_loss
  config.train_config.eval_step_func = eval_steps.eval_step_simple
  config.train_config.optimizer = optax.adam
  config.train_config.batch_size = 8
  config.train_config.epochs = 100
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (224, 224)
  config.model_config.patch_size = (4, 4)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (3, 6, 12, 24)
  config.model_config.window_size = 7
  config.model_config.mlp_ratio = 4.
  config.model_config.qkv_bias = True
  config.model_config.qk_scale = None
  config.model_config.drop_rate = 0.
  config.model_config.attn_drop_rate = 0.
  config.model_config.drop_path_rate = 0.1
  config.model_config.norm_layer = nn.LayerNorm
  config.model_config.ape = False
  config.model_config.patch_norm = True
  config.model_config.use_checkpoint = False
  config.model_config.final_upsample = "expand_first"


  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.0001

  config.augment_config = ml_collections.ConfigDict()
  config.augment_config.aug_function = augmentation_functions.base_custom_randaugment
  config.augment_config.m = 22
  config.augment_config.n = 5

  hash_ = utils.dict_hash(config.to_dict())
  config.group_name = f"base_swin_unet_{hash_}"

  return config


def get_augment_config_2():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.pad_crop_function = jnp.pad
  config.general_config.pad_crop_kwargs = {"pad_width": ((0, 0), (12, 12), (12, 12), (0, 0)), "mode": "constant", "constant_values": 0}
  config.general_config.call_kwargs = {"x": jnp.ones((1, 224, 224, 2))}
  config.general_config.key_rngs = {"drop_path": random.PRNGKey(0)}

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.train_step_func = train_steps.train_step_dsc_loss
  config.train_config.eval_step_func = eval_steps.eval_step_simple
  config.train_config.optimizer = optax.adam
  config.train_config.batch_size = 8
  config.train_config.epochs = 120
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (224, 224)
  config.model_config.patch_size = (4, 4)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (3, 6, 12, 24)
  config.model_config.window_size = 7
  config.model_config.mlp_ratio = 4.
  config.model_config.qkv_bias = True
  config.model_config.qk_scale = None
  config.model_config.drop_rate = 0.
  config.model_config.attn_drop_rate = 0.
  config.model_config.drop_path_rate = 0.1
  config.model_config.norm_layer = nn.LayerNorm
  config.model_config.ape = False
  config.model_config.patch_norm = True
  config.model_config.use_checkpoint = False
  config.model_config.final_upsample = "expand_first"


  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.0002

  config.augment_config = ml_collections.ConfigDict()
  config.augment_config.aug_function = augmentation_functions.base_custom_randaugment
  config.augment_config.m = 22
  config.augment_config.n = 6

  hash_ = utils.dict_hash(config.to_dict())
  config.group_name = f"base_swin_unet_{hash_}"

  return config