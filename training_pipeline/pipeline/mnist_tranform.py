
import tensorflow_transform as tft

IMAGE_KEY = 'image_floats'
LABEL_KEY = 'image_class'


def transformed_name(key):
  return key + '_xf'

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # The input float values for the image encoding are in the range [-0.5, 0.5].
  # So scale_by_min_max is a identity operation, since the range is preserved.
  outputs[transformed_name(IMAGE_KEY)] = (
      tft.scale_by_min_max(inputs[IMAGE_KEY], -0.5, 0.5))
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
  return outputs
