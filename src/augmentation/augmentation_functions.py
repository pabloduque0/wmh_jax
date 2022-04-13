from imgaug import parameters as iap
import imgaug.augmenters as iaa
import numpy as np
import jax.numpy as jnp

_SEED = 42

def _float_parameter(level, maxval):
  maxval_norm = maxval / 30
  return iap.Multiply(level, maxval_norm, elementwise=True)

def _int_parameter(level, maxval):
  # paper applies just int(), so we don't round here
  return iap.Discretize(_float_parameter(level, maxval),
                        round=False)

def no_augmentation(images, labels, n, m):
  return images, labels


def base_custom_randaugment(images, labels, n, m):
  images_dtype, labels_dtype = images.dtype, labels.dtype
  # numpy intentionally here
  images, labels = np.squeeze(np.array(images), axis=0), np.squeeze(np.array(labels), axis=0)
  mandatory_augmentations = iaa.Sequential([
    iaa.Fliplr(0.33, seed=_SEED),
    iaa.Flipud(0.33, seed=_SEED),
    iaa.KeepSizeByResize(
        iaa.Crop(
            percent=iap.Divide(
                iap.Uniform(0, m),
                images.shape[1],
                elementwise=True),
            sample_independently=True,
            keep_size=False,
            seed=_SEED),
        interpolation="linear",
        seed=_SEED)])
  extra_augmentations = iaa.SomeOf(
    n=n,
    children=[
      iaa.meta.Identity(),
      iaa.ScaleX(iap.RandomSign(iap.Uniform(0.9, 1.1)), seed=_SEED),
      iaa.ScaleY(iap.RandomSign(iap.Uniform(0.9, 1.1)), seed=_SEED),
      iaa.Multiply(iap.Uniform(0.7, 1.3), per_channel=False, seed=_SEED),
      iaa.TranslateX(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.TranslateY(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.ShearX(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.ShearY(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.Rotate(iap.RandomSign(_float_parameter(m, 30)), seed=_SEED),
      #iaa.Cutout(1, size=iap.Clip(_float_parameter(m, 20 / 32), 0, 20 / 32),
      #                        squared=True,
      #                        fill_mode="constant",
      #                        cval=0)
      ])
  aug_data, aug_labels = mandatory_augmentations(images=images, segmentation_maps=labels.astype(bool))
  aug_data, aug_labels = extra_augmentations(images=aug_data.astype(np.float32),
                                             segmentation_maps=aug_labels.astype(bool))
  return (jnp.expand_dims(jnp.array(aug_data).astype(images_dtype), axis=0),
          jnp.expand_dims(jnp.array(aug_labels).astype(labels_dtype), axis=0))

def base_custom_randaugment_2(images, labels, n, m):
  images_dtype, labels_dtype = images.dtype, labels.dtype
  mandatory_augmentations = iaa.Sequential([
    iaa.Fliplr(0.33, seed=_SEED),
    iaa.Flipud(0.33, seed=_SEED),
    iaa.KeepSizeByResize(
        iaa.Crop(
            percent=iap.Divide(
                iap.Uniform(0, m),
                images.shape[1],
                elementwise=True),
            sample_independently=True,
            keep_size=False,
            seed=_SEED),
        interpolation="linear",
        seed=_SEED)])
  extra_augmentations = iaa.SomeOf(
    n=n,
    children=[
      iaa.meta.Identity(),
      iaa.ScaleX(iap.RandomSign(iap.Uniform(0.9, 1.1)), seed=_SEED),
      iaa.ScaleY(iap.RandomSign(iap.Uniform(0.9, 1.1)), seed=_SEED),
      iaa.Multiply(iap.Uniform(0.7, 1.3), per_channel=False, seed=_SEED),
      iaa.TranslateX(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.TranslateY(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.ShearX(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.ShearY(iap.RandomSign(_float_parameter(m, 0.33)), seed=_SEED),
      iaa.Rotate(iap.RandomSign(_float_parameter(m, 30)), seed=_SEED),
      iaa.geometric.ElasticTransformation(alpha=100, sigma=100 / _float_parameter(m, 18), seed=_SEED)
      #iaa.Cutout(1, size=iap.Clip(_float_parameter(m, 20 / 32), 0, 20 / 32),
      #                        squared=True,
      #                        fill_mode="constant",
      #                        cval=0)
      ])
  aug_data, aug_labels = mandatory_augmentations(images=images, segmentation_maps=labels.astype(bool))
  aug_data, aug_labels = extra_augmentations(images=aug_data.astype(np.float32),
                                             segmentation_maps=aug_labels.astype(bool))
  return jnp.array(aug_data).astype(images_dtype), jnp.array(aug_labels).astype(labels_dtype)