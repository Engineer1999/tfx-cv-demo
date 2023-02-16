import os
from absl import logging

from tfx import v1 as tfx
from pipeline import mnist_pipeline
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
_metadata_path = config["tfx_config"]["_metadata_path"]


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
