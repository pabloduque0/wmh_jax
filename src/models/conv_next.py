import functools
from typing import Any, Callable, Optional, Sequence, Tuple
from flax.linen import activation

import jax.numpy as jnp
import jax
from flax import linen as nn
"""
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = jnp.transpose(x, axes=(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = jnp.transpose(x, axes=(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
"""

class Block(nn.Module):
  dim: int
  drop_path: float = 0.
  layer_scale_init_value: float = 1e-6
  
  def setup(self):
    self.dwconv = nn.Conv(self.dim, kernel_size=7, padding=3, groups=self.dim) # depthwise conv
    self.norm = nn.LayerNorm(self.dim, eps=1e-6)
    self.pwconv1 = nn.Dense(4 * self.dim) # pointwise/1x1 convs, implemented with linear layers
    self.act = nn.gelu
    self.pwconv2 = nn.Dense(self.dim)
    # self.param
    self.gamma = nn.Parameter(self.layer_scale_init_value * torch.ones((self.dim)), 
                                requires_grad=True) if self.layer_scale_init_value > 0 else None
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
  
  @nn.compact
  def __call__(self, x):
    input = x
    x = self.dwconv(x)
    x = jnp.transpose(x, axes=(0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)
    if self.gamma is not None:
        x = self.gamma * x
    x = jnp.transpose(x, axes=(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)

    x = input + self.drop_path(x)
    return x

class ConvNeXt(nn.Module):
  in_chans=3
  num_classes=1000
  depths=[3, 3, 9, 3]
  dims=[96, 192, 384, 768]
  drop_path_rate=0.
  layer_scale_init_value=1e-6
  head_init_scale=1.

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