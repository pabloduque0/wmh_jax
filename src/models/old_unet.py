import functools
from typing import Any, Callable, Optional, Sequence, Tuple
from flax.linen import activation

import jax.numpy as jnp
import jax
from flax import linen as nn

class UnetGatingSignal(nn.Module):

  @nn.compact
  def __call__(self, x, is_training):
    x = nn.Conv(features=x.shape[3] * 2, kernel_size=(1, 1))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    return x

class AttnGatingBlock(nn.Module):
  features: int
  kernel_size: Tuple[int, int]

  def setup(self):
    
    self.strided_conv_x = nn.Conv(features=self.features, strides=(2, 2), kernel_size=self.kernel_size)
    self.one_by_one_conv_g = nn.Conv(features=self.features, kernel_size=(1, 1))
    self.trans_conv_g = nn.ConvTranspose(features=self.features, kernel_size=self.kernel_size)
    self.mask_conv = nn.Conv(features=1, kernel_size=(1, 1))
    self.output_conv = nn.Conv(features=2, kernel_size=(1, 1))
  
  @nn.compact
  def __call__(self, for_concat_x, from_below_g, is_training):
    x = self.strided_conv_x(for_concat_x)
    g = self.one_by_one_conv_g(from_below_g)
    g = self.trans_conv_g(g)

    added = g + x
    added = nn.activation.relu(added)
    mask = self.mask_conv(added)
    mask = nn.activation.sigmoid(mask)
    upsampled_mask = jax.image.resize(image=mask, shape=for_concat_x.shape, method="nearest")
    
    output = upsampled_mask * for_concat_x
    output = self.output_conv(output)
    output = nn.normalization.BatchNorm(use_running_average=not is_training)(output)
    return output

class FullAttnGateUpConcatBlock(nn.Module):

  features: int
  kernel_size: Tuple[int, int]
  attn_gating_features: int
  attn_gating_kernel_size: Tuple[int, int]
  
  def setup(self):
    self.gating_signal = UnetGatingSignal()
    self.attn_gating = AttnGatingBlock(features=self.attn_gating_features, kernel_size=self.attn_gating_kernel_size)
    self.transposed_conv = nn.ConvTranspose(features=self.features, kernel_size=self.kernel_size, strides=(2, 2))

  def __call__(self, for_concat_x, from_below_g, is_training):
    gating = self.gating_signal(from_below_g, is_training=is_training)
    attention = self.attn_gating(for_concat_x=for_concat_x, from_below_g=from_below_g, is_training=is_training)
    up_samp = self.transposed_conv(from_below_g)
    up_samp = nn.activation.relu(up_samp)
    concat = jnp.concatenate([up_samp, attention], axis=3)
    return concat

class ConvPoolBlock(nn.Module):
  features: int 
  kernel_size: Tuple[int, int]
  activation: Callable[[Any], Any] = nn.relu
  window_shape: Tuple[int, int] = (2, 2)

  def setup(self):
    self.conv1 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
    self.conv2 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
    
  @nn.compact
  def __call__(self, x):
    x = self.conv1(x)
    x = self.activation(x)
    x = self.conv2(x)
    x_for_concat = self.activation(x)
    x = nn.max_pool(x_for_concat, window_shape=self.window_shape, strides=self.window_shape)
    return x, x_for_concat

class MultiConvBlock(nn.Module):
  features: int 
  kernel_size: Tuple[int, int]
  n_convolutions: int
  activation: Callable[[Any], Any] = nn.relu
  padding: str = "SAME"

  def setup(self):
    self.convolutions = [
        nn.Conv(features=self.features,
                kernel_size=self.kernel_size,
                padding=self.padding) for _ in range(self.n_convolutions)
      ]

  @nn.compact
  def __call__(self, x):
    for conv in self.convolutions:
      x = conv(x)
    return x


class CustomAttnGatedUnet(nn.Module):
  stack_features: int
  stack_kernel_size: Tuple[int, int]
  stack_n_convolutions: int

  down_features: Sequence[int]
  down_kernel_size: Sequence[Tuple[int, int]]
  down_window_shape: Sequence[Tuple[int, int]]
  n_steps: int

  bottom_features: Sequence[int]
  bottom_kernel_size: Sequence[Tuple[int, int]]
  n_bottom_convs: int

  up_features: Sequence[int]
  up_kernel_size: Sequence[Tuple[int, int]]
  attn_gating_features: Sequence[int]
  attn_gating_kernel_size: Sequence[Tuple[int, int]]

  last_stack_features: Sequence[int]
  last_stack_kernel_size: Sequence[Tuple[int, int]]
  last_stack_n_convolutions: int

  def setup(self):
    self.conv_stack = MultiConvBlock(features=self.stack_features, kernel_size=self.stack_kernel_size, n_convolutions=self.stack_n_convolutions)
    self.convs_pools = [
        ConvPoolBlock(features=self.down_features[i],
                      kernel_size=self.down_kernel_size[i],
                      window_shape=self.down_window_shape[i])
        for i in range(self.n_steps)
    ]
    self.bottom_convs = [nn.Conv(features=self.bottom_features[i], kernel_size=self.bottom_kernel_size[i]) for i in range(self.n_bottom_convs)]

    self.step_ups = [
        FullAttnGateUpConcatBlock(features=self.up_features[i],
                                  kernel_size=self.up_kernel_size[i],
                                  attn_gating_features=self.attn_gating_features[i],
                                  attn_gating_kernel_size=self.attn_gating_kernel_size[i])
        for i in range(self.n_steps)
    ]
    self.output_convs = MultiConvBlock(features=self.last_stack_features, kernel_size=self.last_stack_kernel_size, n_convolutions=self.last_stack_n_convolutions)
    self.last_conv = nn.Conv(features=1, kernel_size=(1, 1))

  def __call__(self, x, is_training):
    
    x = self.conv_stack(x)
    xs_for_concat = []
    for conv_pool in self.convs_pools:
      x, x_for_concat = conv_pool(x)
      xs_for_concat.append(x_for_concat)

    for conv in self.bottom_convs:
      x = conv(x)

    for step_up, x_for_concat in zip(self.step_ups, reversed(xs_for_concat)):
      x = step_up(x_for_concat, x, is_training=is_training)

    x = self.last_conv(x)
    x = nn.activation.sigmoid(x)
    return x