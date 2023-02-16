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


_pipeline_name = "mnist_native_keras"
_mnist_root = os.path.join(
    "/home/bhargavpatel/Desktop/Image_Classification_tfx/Code_Tinkaring/",
    "image-classification-pipeline",
)
_data_root = os.path.join(_mnist_root, "tfrecords")
_serving_model_dir = os.path.join(_mnist_root, "serving_model", _pipeline_name)
_tfx_root = os.path.join(_mnist_root, "tfx")
_pipeline_root = os.path.join(_tfx_root, "pipelines", _pipeline_name)
_metadata_path = os.path.join(_tfx_root, "metadata", _pipeline_name, "metadata.db")
_transform_module_file = "/home/bhargavpatel/Desktop/Image_Classification_tfx/Code_Tinkaring/image-classification-pipeline/training_pipeline/pipeline/mnist_tranform.py"
trainer_module = "/home/bhargavpatel/Desktop/Image_Classification_tfx/Code_Tinkaring/image-classification-pipeline/training_pipeline/pipeline/mnist_train.py"

_beam_pipeline_args = [
    "--direct_running_mode=multi_processing",
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    "--direct_num_workers=0",
]

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    "schedule_interval": None,
    "start_date": datetime(2019, 1, 1),
}


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
        module_file=trainer_module,
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
