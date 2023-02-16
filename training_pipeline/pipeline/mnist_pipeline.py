import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
import tfx
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import BulkInferrer
from tfx.proto import bulk_inferrer_pb2
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from datetime import datetime

import tomli

_config_path = "config.toml"


def read_config(path):
    with open(path) as f:
        content = f.read()
        config = tomli.loads(content)

    return config


config = read_config(_config_path)

_pipeline_name = config["tfx_config"]["_pipeline_name"]
_data_root = config["tfx_config"]["_data_root"]
_serving_model_dir = config["tfx_config"]["_serving_dir"]
_pipeline_root = config["tfx_config"]["_pipeline_root"]
_transform_module_file = config["tfx_config"]["_transform_module_file"]
_trainer_module = config["tfx_config"]["_trainer_module"]


def _create_pipeline(
    pipeline_name: str, pipeline_root: str, data_root: str, serving_model_dir: str, metadata_path: str
) -> pipeline.Pipeline:

    example_gen = ImportExampleGen(input_base=_data_root)
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    infer_schema = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"], schema=infer_schema.outputs["schema"]
    )
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=infer_schema.outputs["schema"],
        module_file=_transform_module_file,
    )
    trainer = Trainer(
        module_file=_trainer_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=infer_schema.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(num_steps=100),
    )
    accuracy_threshold = 0.8
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="image_class")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="SparseCategoricalAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(lower_bound={"value": accuracy_threshold})
                        ),
                    )
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"], model=trainer.outputs["model"], eval_config=eval_config
    )
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=_serving_model_dir)
        ),
    )
    bulk_inferrer = BulkInferrer(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        data_spec=bulk_inferrer_pb2.DataSpec(),
        model_spec=bulk_inferrer_pb2.ModelSpec(),
    )

    return pipeline.Pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            infer_schema,
            validate_stats,
            transform,
            trainer,
            evaluator,
            pusher,
            bulk_inferrer,
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
    )
