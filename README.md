# MNIST Classification Pipeline using TFX

This repository shows how to build a Machine Learning Pipeline for Image Classification with Tensorflow Extended, Docker, and AWS. The main focus of this repo is to show how to use TFX and Docker to build the pipelie rather then what is image classification.

# Project Structure
```bash
project
│
└───tfrecords
│   │   mnist.tfrecord # MNIST dataset converted in tfrecords
│
└───training_pipeline 
│   │───pipeline
│   │   └─── mnist_pipeline.py  # TFX pipeline
│   │   └─── mnist_train.py     # Training functions
│   │   └─── mnist_transform.py # Data preprocessing function
│   └─── local_runner.py        # Python code for triggering pipeline
│
└───config.toml            # TOML config file for env variables
└───Dockerfile             # To build docker image
└───Makefile               # To run a sequence of bash commands
└───Requirement.txt        # List of python library
└───upload_model_to_s3.py  # To upload trained model to S3
```

# Instructions

The TFX pipeline is designed to be run on both of local environments, with minor changes in code, it can run on GCP with Kubeflow or Vertex AI

## On Local Environment
```
$ cd tfx-cv-demo

$ tfx pipeline create --pipeline-path=training_pipeline/local_runner.py \
                      --engine=local
$ tfx pipeline compile --pipeline-path=training_pipeline/local_runner.py \
                       --engine=local
$ tfx run create --pipeline-name=mnist_native_keras_docker \ 
                 --engine=local
$ python upload_model_to_s3.py

$ rm -rf serving_model
```

## OR

```
$ make run_pipeline
```

# Docker
To build the docker image and run it, follow below steps

## Build Docker image
```
$ cd tfx-cv-demo

$ docker build -t tfx_cv_demo:1.0.0 .

$ docker run  \
       --env AWS_ACCESS_KEY_ID=<aws_access_key> \
       --env AWS_SECRET_ACCESS_KEY=<aws_secret_access_key> \ 
       -it <image_id>

$ docker login -u <user_id> -p <password>

$ docker tag tfx_cv_demo:1.0.0 <user_id>/<repo_name>

$ docker push <user_id>/<repo_name>
```