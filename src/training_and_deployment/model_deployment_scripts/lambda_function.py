# This python script can be deployed as a lambda function handler in
# the same account/region as the SMEP in order to test the model endpoint.

import json
import boto3

def lambda_handler(event, context):
    # Initialize the SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')

    # Specify endpoint name
    endpoint_name = 'gpt2l-v01-endpoint'

    # Prepare the input data
    input_data = json.dumps({
        "text": "Is a Ferrari a fast type of car?"
    })

    try:
        # Invoke the endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())

        # optionally slice off everything after "Response"
        # result = result.split("Response")[1]

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        print(f"Error invoking endpoint: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error invoking endpoint: {str(e)}")
        }