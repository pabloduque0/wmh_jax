import ml_collections
import flax.linen as nn
from src.models import metrics
from src.augmentation import augmentation_functions
from src.train import eval_steps
from src.train import train_steps
import optax

def get_initial_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()
  config.group_name = "base_multi_scale_no_aug"
  config.general_config = ml_collections.ConfigDict()
  config.general_config.model_name = "MultiScaleUnet"
  config.general_config.data_path = "data/processed/noskull_stand_noaug"
  config.general_config.multi_label = True
  #config.general_config.levels_multilabel = 3

  config.train_config = ml_collections.ConfigDict()
  config.train_config.train_step_func = train_steps.train_step_bn_multi_scale_dsc_loss
  config.train_config.eval_step_func = eval_steps.eval_step_multi_scale_bn
  config.train_config.optimizer = optax.adam
  config.train_config.batch_size = 2
  config.train_config.epochs = 100
  
  config.model_config = ml_collections.ConfigDict()
  config.model_config.conv_block_features = (64, 96, 128, 256)
  config.model_config.conv_block_kernel_sizes = ((5, 5), (3, 3), (3, 3), (3, 3))
  config.model_config.conv_block_window_shape = ((2, 2), (2, 2), (2, 2), (2, 2))
  config.model_config.conv_block_pool_strides = ((2, 2), (2, 2), (2, 2), (2, 2))
  
  config.model_config.convup_block_features = (512, 256, 128, 96)
  config.model_config.convup_block_kernel_size = ((3, 3), (3, 3), (3, 3), (3, 3))

  config.model_config.last_conv_features = 64
  config.model_config.last_conv_kernel_size = (3, 3)

  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.learning_rate = 0.0002

  config.augment_config = ml_collections.ConfigDict()
  config.augment_config.aug_function = augmentation_functions.no_augmentation
  config.augment_config.m = None
  config.augment_config.n = None

  return config