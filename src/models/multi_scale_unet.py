import functools
from typing import Any, Callable, Optional, Sequence, Tuple
from flax.linen import activation

import jax.numpy as jnp
import jax
from flax import linen as nn

class ConvBNLayer(nn.Module):
  features: int
  kernel_size: Tuple[int, int]

  def setup(self):
    self.conv = nn.Conv(features=self.features,
                        kernel_size=self.kernel_size)

  @nn.compact
  def __call__(self, x, is_training):
    x = self.conv(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    return nn.activation.elu(x)


class ConvPoolBlock(nn.Module):

  features: int
  kernel_size: Tuple[int, int]
  window_shape: Tuple[int, int]
  pool_strides: Tuple[int, int]

  def setup(self):
    self.conv1 = ConvBNLayer(features=self.features, kernel_size=self.kernel_size)
    self.conv2 = ConvBNLayer(features=self.features, kernel_size=self.kernel_size)
    

  def __call__(self, x, is_training):
    x = self.conv1(x, is_training=is_training)
    x_for_concat = self.conv2(x, is_training=is_training)
    x = nn.max_pool(inputs=x_for_concat,
                    window_shape=self.window_shape,
                    strides=self.pool_strides,
                    padding="SAME")
    return x, x_for_concat


class ConvUpBlock(nn.Module):
  features: int
  kernel_size: Tuple[int, int]

  def setup(self):
    self.conv1 = ConvBNLayer(features=self.features, kernel_size=self.kernel_size)
    self.conv2 = ConvBNLayer(features=self.features, kernel_size=self.kernel_size)    

  def __call__(self, x, x_for_concat, is_training):
    B, H, W, C = x.shape
    x = self.conv1(x, is_training=is_training)
    x_for_output = self.conv2(x, is_training=is_training)
    new_shape = (B, x_for_concat.shape[1], x_for_concat.shape[2], C)
    x = jax.image.resize(x_for_output, shape=new_shape, method="nearest")
    x = jnp.concatenate([x, x_for_concat], axis=-1)
    return x, x_for_output


class OutputBlock(nn.Module):

  def setup(self):
    self.conv = nn.Conv(features=1, kernel_size=(1, 1))

  def __call__(self, x):
    x = self.conv(x)
    x = nn.activation.sigmoid(x)
    return x

class MultiScaleUnet(nn.Module):
  
  conv_block_features: Sequence[int] 
  conv_block_kernel_sizes: Sequence[int]
  conv_block_window_shape: Sequence[Tuple[int, int]]
  conv_block_pool_strides: Sequence[Tuple[int, int]]

  convup_block_features: Sequence[int]
  convup_block_kernel_size: Sequence[Tuple[int, int]]

  last_conv_features: int
  last_conv_kernel_size: Tuple[int, int]

  def setup(self):
    
    self.conv_pools = [
      ConvPoolBlock(features=feats, kernel_size=k_size, window_shape=window, pool_strides=pool_stride)
      for feats, k_size, window, pool_stride in zip(self.conv_block_features,
                                                    self.conv_block_kernel_sizes, 
                                                    self.conv_block_window_shape,
                                                    self.conv_block_pool_strides)
    ]
    self.conv_upsamps = [
      ConvUpBlock(features=feats, kernel_size=k_size)
      for feats, k_size in zip(self.convup_block_features,
                               self.convup_block_kernel_size)
    ]
    self.output_layers = [OutputBlock() for _ in range(len(self.convup_block_features))]
    self.last_conv1 = ConvBNLayer(features=self.last_conv_features, kernel_size=self.last_conv_kernel_size)
    self.last_conv2 = ConvBNLayer(features=self.last_conv_features, kernel_size=self.last_conv_kernel_size)
    self.main_output_layer = OutputBlock()

  def __call__(self, x, is_training):

    xs_for_concat = []
    for conv_pool in self.conv_pools:
      x, x_for_concat = conv_pool(x, is_training=is_training)
      xs_for_concat.append(x_for_concat)

    xs_for_output = []
    for step_up, x_for_concat in zip(self.conv_upsamps, reversed(xs_for_concat)):
      x, x_output = step_up(x=x,
                            x_for_concat=x_for_concat,
                            is_training=is_training)
      xs_for_output.append(x_output)

    sub_outputs = []
    for x_output, output_layer in zip(xs_for_output, self.output_layers):
      sub_output = output_layer(x_output)
      sub_outputs.append(sub_output)

    x = self.last_conv1(x, is_training=is_training)
    x = self.last_conv2(x, is_training=is_training)
    x = self.main_output_layer(x)
    return x, *sub_outputs[::-1]