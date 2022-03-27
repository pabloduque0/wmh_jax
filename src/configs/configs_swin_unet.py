import ml_collections
import flax.linen as nn
from src.models import metrics

def get_initial_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.group_name = "base_swin_unet"
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = False
  config.general_config.levels_multilabel = 3

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.learning_rate = 0.00001
  config.train_config.batch_size = 16
  config.train_config.epochs = 2
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (192, 192)
  config.model_config.patch_size = (4, 4)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (3, 6, 12, 24)
  config.model_config.window_size = 6
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

  return config


def get_param_increase_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.group_name = "param_increase"
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand"
  config.general_config.multi_label = False
  config.general_config.levels_multilabel = 3

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.learning_rate = 0.00001
  config.train_config.batch_size = 25
  config.train_config.epochs = 45
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (192, 192)
  config.model_config.patch_size = (12, 12)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (6, 12, 24, 48)
  config.model_config.window_size = 16
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

  return config

def get_param_increase_new_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.group_name = "param_increase_new_test"
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "SwinUnet"
  config.general_config.data_path = "data/processed/noskull_stand"
  config.general_config.multi_label = False
  config.general_config.levels_multilabel = 3

  config.train_config = ml_collections.ConfigDict()
  #config.train_config.loss_function = metrics.dice_loss
  config.train_config.learning_rate = 0.00001
  config.train_config.batch_size = 25
  config.train_config.epochs = 45
  #config.train_config.metrics_to_calc = (metrics.dice_coefficient, )
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.img_size = (192, 192)
  config.model_config.patch_size = (4, 4)
  config.model_config.in_chans = 2
  config.model_config.num_classes = 1
  config.model_config.embed_dim = 96
  config.model_config.depths = (2, 2, 2, 2)
  config.model_config.depths_decoder = (1, 2, 2, 2)
  config.model_config.num_heads = (6, 12, 24, 48)
  config.model_config.window_size = 6
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

  return config