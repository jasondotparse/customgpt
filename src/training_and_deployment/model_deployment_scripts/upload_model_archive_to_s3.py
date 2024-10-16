import boto3
from datetime import datetime

current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

s3 = boto3.resource('s3')

s3.meta.client.upload_file(
    "./model_files_for_s3/model_archive.tar.gz",
    "custom-gpt-artifacts",
    f"{current_datetime}/model_archive.tar.gz"
)