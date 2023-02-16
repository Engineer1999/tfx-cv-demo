import os
from absl import logging

from tfx import v1 as tfx
from pipeline import mnist_pipeline
from datetime import datetime
import tomli


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
# _beam_pipeline_args = [
#     "--direct_running_mode=multi_processing",
#     # 0 means auto-detect based on on the number of CPUs available
#     # during execution time.
#     "--direct_num_workers=0",
# ]

# # Airflow-specific configs; these will be passed directly to airflow
# _airflow_config = {
#     "schedule_interval": None,
#     "start_date": datetime(2019, 1, 1),
# }


def run():
    tfx.orchestration.LocalDagRunner().run(
        mnist_pipeline._create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
