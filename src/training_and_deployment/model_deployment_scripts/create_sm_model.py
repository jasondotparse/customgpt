import boto3
import sagemaker
from datetime import datetime

sagemaker_client = boto3.client(service_name="sagemaker")
role = sagemaker.get_execution_role()

current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

model_name = f"custom-gpt-{current_datetime}"
my_aws_account = "079556679466"
my_aws_region = "us-west-2"
my_s3_bucket = "model-for-toucan-jasholtk-dev"
model_s3_object_name = "customGPT/model_archive.tar.gz"
ecr_repo_name = "jasholtk/llm"

primary_container = {
    "Image": f"{my_aws_account}.dkr.ecr.{my_aws_region}.amazonaws.com/{ecr_repo_name}:latest",
    "ModelDataUrl": f"s3://{my_s3_bucket}/{model_s3_object_name}"
}

create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=primary_container)