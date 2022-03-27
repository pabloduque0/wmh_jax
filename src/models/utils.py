from jax import scipy as jscipy
import jax
import jax.numpy as jnp
from flax import linen as nn


def truncated_normal(rng, shape, mean=0., std=1., min_cutoff=-2., max_cutoff=2.):
  def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + jscipy.special.erf(x / jnp.sqrt(2.))) / 2.
  l = norm_cdf((min_cutoff - mean) / std)
  u = norm_cdf((max_cutoff - mean) / std)
  x = jax.random.uniform(rng, minval=2 * l - 1, maxval=2 * u - 1, shape=shape)
  x = jscipy.special.erfinv(x)
  x *= std * jnp.sqrt(2.)
  x += mean
  x = jax.lax.clamp(min=min_cutoff, x=x, max=max_cutoff)
  return x


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
      super().__init__()
      
    def __call__(self, x):
        return x
