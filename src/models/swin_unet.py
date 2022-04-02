import functools
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import jax
from flax import linen as nn
import einops
import time
from src.models import utils

class MlpBlock(nn.Module):
  out_features: int
  hidden_features: int
  dropout_rate: float = 0.
  
  def setup(self):
    self.fc1 = nn.Dense(self.hidden_features)
    self.fc2 = nn.Dense(self.out_features)
    self.drop = nn.Dropout(self.dropout_rate, deterministic=True)
    
  @nn.compact
  def __call__(self, x):
    pre = time.time()
    x = self.fc1(x)
    x = nn.gelu(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x

def _get_relative_position_index(window_size):
    # get pair-wise relative position index for each token inside the window
    coords_h = jnp.arange(window_size[0])
    coords_w = jnp.arange(window_size[1])
    coords = jnp.stack(jnp.meshgrid(coords_w, coords_h, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = coords.reshape((2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = jnp.transpose(relative_coords, [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
    relative_coords = jax.ops.index_add(relative_coords, jax.ops.index[:, :, 0], 4 - 1)  # shift to start from 0
    relative_coords = jax.ops.index_add(relative_coords, jax.ops.index[:, :, 1], 4 - 1)
    relative_coords = jax.ops.index_mul(relative_coords, jax.ops.index[:, :, 0], 2 * 4 - 1)
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


class WindowAttention(nn.Module):
  dimension: int
  window_size: Tuple[int, int] # Wh, Ww
  num_heads: int
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  attn_drop_val: float = 0.
  proj_drop_val: float = 0.

  def setup(self):
    self.head_dim = self.dimension // self.num_heads
    self.qkv = nn.Dense(self.dimension * 3)#, bias=self.qkv_bias)
    self.scale = self.qk_scale or self.head_dim ** -0.5
    self.attn_drop = nn.Dropout(self.attn_drop_val, deterministic=False)
    self.proj = nn.Dense(self.dimension)
    self.proj_drop = nn.Dropout(self.proj_drop_val, deterministic=False)
    self.relative_position_bias_table_ = self.param(
        'relative_position_bias_table',
        utils.truncated_normal,
        ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
    )# 2*Wh-1 * 2*Ww-1, nH
    """
    self.relative_position_index = self.variable("aux_variables", "relative_position_index",
                                                 _get_relative_position_index,
                                                self.window_size)
    """
    self.relative_position_index = _get_relative_position_index(self.window_size)

  def __call__(self, x, mask=None):
    pre = time.time()
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
    qkv = jnp.transpose(qkv, [2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    q = q * self.scale
    attn = (q @ jnp.swapaxes(k, -2, -1))
    
    relative_position_bias = self.relative_position_bias_table_[self.relative_position_index.reshape(-1)].reshape(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
    relative_position_bias = jnp.transpose(relative_position_bias, [2, 0, 1])
    attn = attn + jnp.expand_dims(relative_position_bias, 0)
    
    if mask is not None:
      nW = mask.shape[0]
      attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + jnp.expand_dims(jnp.expand_dims(mask, 1), 0)
      attn = attn.reshape(-1, self.num_heads, N, N)
      attn = nn.softmax(attn)
    
    attn = nn.softmax(attn)
    attn = self.attn_drop(attn)

    x = jnp.swapaxes(attn @ v, 1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    drop_prob: float
    is_training: bool

    def __call__(self, x, rng = None):
      pre = time.time()
      if self.drop_prob == 0. or not self.is_training:
        return x
      keep_prob = 1 - self.drop_prob
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
      if not rng:
        rng = self.make_rng('drop_path')
      random_tensor = jnp.floor(keep_prob + jax.random.uniform(key=rng, shape=shape, dtype=x.dtype))
      output = x / keep_prob * random_tensor
      #print(self.__class__.__name__, time.time() - pre)
      return output


@functools.partial(jax.jit, static_argnums=1)
def window_partition(x, window_size):
  """
  Args:
      x: (B, H, W, C)
      window_size (int): window size
  Returns:
      windows: (num_windows*B, window_size, window_size, C)
  """
  B, H, W, C = x.shape
  x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
  windows = jnp.transpose(x, [0, 1, 3, 2, 4, 5]).reshape(-1, window_size, window_size, C)
  return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = jnp.transpose(x, [0, 1, 3, 2, 4, 5]).reshape(B, H, W, -1)
    return x
  
def _get_attention_mask(shift_size, window_size, input_resolution):
  
  if shift_size > 0:
    # calculate attention mask for SW-MSA
    H, W = input_resolution
    img_mask = jnp.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
      for w in w_slices:
        img_mask = jax.ops.index_update(x=img_mask, idx=jax.ops.index[:, h, w, :], y=cnt)
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.reshape(-1, window_size * window_size)
    attn_mask = jnp.expand_dims(mask_windows, 1) - jnp.expand_dims(mask_windows, 2)
    attn_mask = jnp.where(attn_mask != 0, float(-100.0), attn_mask)
    attn_mask = jnp.where(attn_mask == 0, float(0.0), attn_mask)
  else:
    attn_mask = None
  return attn_mask

class SwinTransformerBlock(nn.Module):
  dimension: int
  input_resolution: Tuple[int, int]
  window_size: int
  num_heads: int
  shift_size: int
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  drop: float = 0.
  attn_drop_val: float = 0.
  drop_path_value: float = 0.
  mlp_ratio: float = 4.
  is_training: bool = True
  norm_layer: nn.Module = nn.LayerNorm
  
  def setup(self):
    self.normalization1 = self.norm_layer()
    self.window_attn = WindowAttention(
            self.dimension, window_size=(self.window_size, self.window_size), num_heads=self.num_heads,
            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop_val=self.attn_drop_val, proj_drop_val=self.drop)
    
    
    
    self.attn_mask = _get_attention_mask(self.shift_size, self.window_size, self.input_resolution)
    self.normalization2 = self.norm_layer()
    mlp_hidden_dim = int(self.dimension * self.mlp_ratio)
    self.mlp = MlpBlock(out_features=self.dimension, hidden_features=mlp_hidden_dim, dropout_rate=self.drop)
    self.drop_path = utils.Identity() #TODO Add actual drop path
    #self.drop_path = DropPath(drop_prob=self.drop_path_value, is_training=self.is_training)
    
    
  @nn.compact
  def __call__(self, x):
    pre = time.time()
    H, W = self.input_resolution
    B, L, C = x.shape
    
    shortcut = x
    x = self.normalization1(x)
    x = x.reshape(B, H, W, C)
    
    if self.shift_size > 0:
      shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
    else:
      shifted_x = x
    
    # partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
    # W-MSA/SW-MSA
    attn_windows = self.window_attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    # merge windows
    attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
    else:
        x = shifted_x
    x = x.reshape(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.normalization2(x)))
    #print(self.__class__.__name__, time.time() - pre)
    return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    input_resolution: Tuple[int, int]
    dimension: int
    norm_layer: nn.Module = nn.LayerNorm
  
    def setup(self):
        self.reduction = nn.Dense(features=2 * self.dimension, use_bias=False)
        self.normalization = self.norm_layer()

    def __call__(self, x):
        """
        x: B, H*W, C
        """
        pre = time.time()
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = jnp.concatenate([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.normalization(x)
        x = self.reduction(x)
        #print(self.__class__.__name__, time.time() - pre)
        return x

class PatchExpand(nn.Module):
    input_resolution: Tuple[int, int]
    dimension: int
    dim_scale: int = 2
    norm_layer: nn.Module = nn.LayerNorm

    def setup(self):
      self.expand = nn.Dense(features=self.dimension * 2, use_bias=False) if self.dim_scale == 2 else utils.Identity()
      self.normalization = self.norm_layer()
    
    @nn.compact
    def __call__(self, x):
      """
      x: B, H*W, C
      """
      pre = time.time()
      H, W = self.input_resolution
      x = self.expand(x)
      B, L, C = x.shape
      assert L == H * W, "input feature has wrong size"

      x = x.reshape(B, H, W, C)
      x = einops.rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
      x = x.reshape(B,-1,C//4)
      x = self.normalization(x)
      #print(self.__class__.__name__, time.time() - pre)
      return x

class FinalPatchExpand_X4(nn.Module):
  input_resolution: int
  dimension: int
  dim_scale: int = 4
  norm_layer: nn.Module = nn.LayerNorm

  def setup(self):
    self.expand = nn.Dense(16*self.dimension, use_bias=False)
    self.output_dim = self.dimension
    self.normalization = self.norm_layer()

  def __call__(self, x):
    """
    x: B, H*W, C
    """
    pre = time.time()
    H, W = self.input_resolution
    x = self.expand(x)
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    x = x.reshape(B, H, W, C)
    x = einops.rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
    x = x.reshape(B,-1,self.output_dim)
    x = self.normalization(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x

class BasicLayerDown(nn.Module):
  """ A basic Swin Transformer layer for one stage.
  Args:
      dim (int): Number of input channels.
      input_resolution (tuple[int]): Input resolution.
      depth (int): Number of blocks.
      num_heads (int): Number of attention heads.
      window_size (int): Local window size.
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
      qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
      qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
      drop (float, optional): Dropout rate. Default: 0.0
      attn_drop (float, optional): Attention dropout rate. Default: 0.0
      drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
      norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
      downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
      use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
  """

  dimension: int
  input_resolution: int
  depth: int
  num_heads: int
  window_size: int
  mlp_ratio: int = 4.
  qkv_bias: bool = True
  qk_scale: float = None
  drop: float = 0.
  attn_drop: float = 0.
  drop_path: float = 0.
  downsample_module: Optional[nn.Module] = None
  norm_layer: Optional[nn.Module] = nn.LayerNorm
  
  def setup(self):

    # build blocks
    self.blocks = [
        SwinTransformerBlock(dimension=self.dimension, input_resolution=self.input_resolution,
                             num_heads=self.num_heads, window_size=self.window_size,
                             shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                             drop=self.drop, attn_drop_val=self.attn_drop,
                             norm_layer=self.norm_layer,
                             drop_path_value=self.drop_path[i] if isinstance(self.drop_path, tuple) else self.drop_path)
        for i in range(self.depth)]

    # patch merging layer
    if self.downsample_module is not None:
        self.downsample = self.downsample_module(self.input_resolution, dimension=self.dimension)
    else:
        self.downsample = None

  def __call__(self, x):
    pre = time.time()
    for blk in self.blocks:
      x = blk(x)
    if self.downsample is not None:
      x = self.downsample(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x


class BasicLayerUp(nn.Module):
  """ A basic Swin Transformer layer for one stage.
  Args:
      dim (int): Number of input channels.
      input_resolution (tuple[int]): Input resolution.
      depth (int): Number of blocks.
      num_heads (int): Number of attention heads.
      window_size (int): Local window size.
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
      qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
      qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
      drop (float, optional): Dropout rate. Default: 0.0
      attn_drop (float, optional): Attention dropout rate. Default: 0.0
      drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
      norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
      downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
  """

  dimension: int
  input_resolution: Tuple[int, int]
  depth: int
  num_heads: int 
  window_size: int
  mlp_ratio: float = 4.
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  drop: float = 0.
  attn_drop: float = 0.
  drop_path: float = 0.
  norm_layer: nn.Module = nn.LayerNorm
  upsample_module: Optional[nn.Module] = None

  def setup(self):
  # build blocks
    self.blocks = [
        SwinTransformerBlock(dimension=self.dimension, input_resolution=self.input_resolution,
                             num_heads=self.num_heads, window_size=self.window_size,
                             shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                             drop=self.drop, attn_drop_val=self.attn_drop,
                             drop_path_value=self.drop_path[i] if isinstance(self.drop_path, tuple) else self.drop_path)
        for i in range(self.depth)]

    # patch merging layer
    if self.upsample_module is not None:
        self.upsample = PatchExpand(self.input_resolution, dimension=self.dimension, dim_scale=2, norm_layer=self.norm_layer)
    else:
        self.upsample = None

  def __call__(self, x):
    pre = time.time()
    for blk in self.blocks:
      x = blk(x)
    if self.upsample is not None:
      x = self.upsample(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x


class PatchEmbed(nn.Module):
  r""" Image to Patch Embedding
  Args:
      img_size (int): Image size.  Default: 224.
      patch_size (int): Patch token size. Default: 4.
      in_chans (int): Number of input image channels. Default: 3.
      embed_dim (int): Number of linear projection output channels. Default: 96.
      norm_layer (nn.Module, optional): Normalization layer. Default: None
  """

  img_size: Tuple[int, int]
  patch_size: Tuple[int, int]

  in_chans: int = 2
  embed_dim: int = 96
  norm_layer: nn.Module = nn.LayerNorm
  
  def setup(self):
    self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
    self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
    self.proj = nn.Conv(features=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size)
    self.norm = self.norm_layer()

  def __call__(self, x):
    pre = time.time()
    B, H, W, C = x.shape
    # FIXME look at relaxing size constraints
    assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x)
    x = x.reshape(B, -1, self.embed_dim)
      # B Ph*Pw C
    if self.norm is not None:
      x = self.norm(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x
  

class SwinTransformerSys(nn.Module):
  r""" Swin Transformer
      A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/pdf/2103.14030
  Args:
      img_size (int | tuple(int)): Input image size. Default 224
      patch_size (int | tuple(int)): Patch size. Default: 4
      in_chans (int): Number of input image channels. Default: 3
      num_classes (int): Number of classes for classification head. Default: 1000
      embed_dim (int): Patch embedding dimension. Default: 96
      depths (tuple(int)): Depth of each Swin Transformer layer.
      num_heads (tuple(int)): Number of attention heads in different layers.
      window_size (int): Window size. Default: 7
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
      qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
      qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
      drop_rate (float): Dropout rate. Default: 0
      attn_drop_rate (float): Attention dropout rate. Default: 0
      drop_path_rate (float): Stochastic depth rate. Default: 0.1
      norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
      ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
      patch_norm (bool): If True, add normalization after patch embedding. Default: True
      use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
  """

  img_size: Tuple[int, int]
  patch_size: Tuple[int, int]
  in_chans: int
  num_classes: int
  embed_dim: int
  depths: Sequence[int]
  depths_decoder: Sequence[int]
  num_heads: Sequence[int]
  window_size: int
  mlp_ratio: float
  qkv_bias: bool
  qk_scale: Optional[float]
  drop_rate: float
  attn_drop_rate: float
  drop_path_rate: float
  norm_layer: float 
  ape: bool
  patch_norm: bool
  use_checkpoint: bool
  final_upsample: str

  def setup(self):
    self.num_layers = len(self.depths)
    self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
    
    self.num_features_up = int(self.embed_dim * 2)
    
    self.patch_embed = PatchEmbed(
        img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
        norm_layer=self.norm_layer if self.patch_norm else None)
    num_patches = self.patch_embed.num_patches
    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution
    
    if self.ape:
      self.absolute_pos_embed = self.param(
        'absolute_pos_embed',
        utils.truncated_normal,
        (1, num_patches, self.embed_dim), std=.02)
      
    self.pos_drop = nn.Dropout(rate=self.drop_rate, deterministic=False)
    
    #dpr = [x.item() for x in jnp.linspace(0, self.drop_path_rate, sum(self.depths))]
    dpr = jnp.linspace(0, self.drop_path_rate, sum(self.depths))

    self.layers_down = [BasicLayerDown(dimension=int(self.embed_dim * 2 ** i_layer),
                             input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                               self.patches_resolution[1] // (2 ** i_layer)),
                             depth=self.depths[i_layer],
                             num_heads=self.num_heads[i_layer],
                             window_size=self.window_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                             drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                             drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                             norm_layer=self.norm_layer,
                             downsample_module=PatchMerging if (i_layer < self.num_layers - 1) else None)
                        for i_layer in range(self.num_layers)]
  # build decoder layers
    layers_up = []
    concat_back_dim = []
    for i_layer in range(self.num_layers):
        concat_linear = nn.Dense(int(self.embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else utils.Identity()
        if i_layer == 0:
            layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                     patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                   dimension=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                   dim_scale=2,
                                   norm_layer=self.norm_layer)
        else:
            layer_up = BasicLayerUp(dimension=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)),
                            input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                            depth=self.depths[(self.num_layers-1-i_layer)],
                            num_heads=self.num_heads[(self.num_layers-1-i_layer)],
                            window_size=self.window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                            drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                            drop_path=dpr[sum(self.depths[:(self.num_layers-1-i_layer)]):sum(self.depths[:(self.num_layers-1-i_layer) + 1])],
                            norm_layer=self.norm_layer,
                            upsample_module=PatchExpand if (i_layer < self.num_layers - 1) else None)
        layers_up.append(layer_up)
        concat_back_dim.append(concat_linear)

    self.layers_up = layers_up
    self.concat_back_dim = concat_back_dim

    self.norm = self.norm_layer()
    self.norm_up= self.norm_layer()
    
    if self.final_upsample == "expand_first":
      
      self.up = FinalPatchExpand_X4(input_resolution=(self.img_size[0]//self.patch_size[0],
                                                      self.img_size[1]//self.patch_size[1]),
                                    dim_scale=4, dimension=self.embed_dim)
      self.output = nn.Conv(features=self.num_classes,
                            kernel_size=(1, 1),
                            use_bias=False)

  #Encoder and Bottleneck
  def forward_down_features(self, x):
    x = self.patch_embed(x)
    if self.ape:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    x_downsample = []

    for layer in self.layers_down:
        x_downsample.append(x)
        x = layer(x)

    x = self.norm(x)  # B L C
    return x, x_downsample

  #Dencoder and Skip connection
  def forward_up_features(self, x, x_downsample):
    for inx, layer_up in enumerate(self.layers_up):
        if inx == 0:
            x = layer_up(x)
        else:
            x = jnp.concatenate([x,x_downsample[len(self.layers_up) - 1 - inx]],-1)
            x = self.concat_back_dim[inx](x)
            x = layer_up(x)

    x = self.norm_up(x)  # B L C

    return x

  def up_x4(self, x):
    H, W = self.patches_resolution
    B, L, C = x.shape
    assert L == H*W, "input features has wrong size"

    if self.final_upsample=="expand_first":
      x = self.up(x)
      x = x.reshape(B,4*H,4*W,-1)
      x = self.output(x)

    return x

  def __call__(self, x):
    pre = time.time()
    x, x_downsample = self.forward_down_features(x)
    x = self.forward_up_features(x,x_downsample)
    x = self.up_x4(x)
    x = nn.sigmoid(x)
    #print(self.__class__.__name__, time.time() - pre)
    return x
