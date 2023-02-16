import boto3
import tomli

import os
from tqdm import tqdm
from datetime import datetime

_config_path = "config.toml"


def read_config(path):
    with open(path) as f:
        content = f.read()
        config = tomli.loads(content)

    return config


config = read_config(_config_path)


model_exported_path = config["aws"]["model_exported_path"] + "/" + config["tfx_config"]["_pipeline_name"]
bucket_name = config["aws"]["model_bucket_name"]
output_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
output_path = config["tfx_config"]["_pipeline_name"] + "_" + output_time
s3 = boto3.client(
    "s3", aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)


def upload_folder_to_s3(s3_client, s3bucket, input_dir, s3_path):
    pbar = tqdm(os.walk(input_dir))
    for path, subdirs, files in pbar:
        for file in files:
            dest_path = path.replace(input_dir, "").replace(os.sep, "/")
            s3_file = f"{s3_path}/{dest_path}/{file}".replace("//", "/")
            local_file = os.path.join(path, file)
            s3_client.upload_file(local_file, s3bucket, s3_file)
            pbar.set_description(f"Uploaded {local_file} to {s3_file}")
    print(f"Successfully uploaded {input_dir} to S3 {s3_path}")


# Upload the saved model directory to S3
upload_folder_to_s3(s3, bucket_name, model_exported_path, output_path)

