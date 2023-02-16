
from typing import List
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import FnArgs
import absl
import tomli

_config_path = "config.toml"


def read_config(path):
    with open(path) as f:
        content = f.read()
        config = tomli.loads(content)

    return config


config = read_config(_config_path)
IMAGE_KEY = 'image_floats'
LABEL_KEY = 'image_class'


def transformed_name(key):
  return key + '_xf'


def build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying MNIST data.
  Returns:
    A Keras Model.
  """
  # The model below is built with Sequential API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.InputLayer(
          input_shape=(784,), name=transformed_name(IMAGE_KEY)))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=config["hyperperameter"]["_learning_rate"]),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)
  return model

def input_fn(file_pattern: List[str],
             data_accessor: DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.
  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch
  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema).repeat()

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn

# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.
  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(fn_args.train_files, fn_args.data_accessor,
                                tf_transform_output, 40)
  eval_dataset = input_fn(fn_args.eval_files, fn_args.data_accessor,
                               tf_transform_output, 40)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = build_keras_model()

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
