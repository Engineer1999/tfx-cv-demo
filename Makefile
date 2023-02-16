run_pipeline:
	tfx pipeline create --pipeline-path=training_pipeline/local_runner.py --engine=local
	tfx pipeline compile --pipeline-path=training_pipeline/local_runner.py --engine=local
	tfx run create --pipeline-name=mnist_native_keras_docker --engine=local
	python upload_model_to_s3.py
	rm -rf serving_model
