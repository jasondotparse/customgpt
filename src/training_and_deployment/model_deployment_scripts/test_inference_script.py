# from sagemaker import image_uris
# image_uris.retrieve(framework='pytorch',region='us-west-2',version='1.8.0',py_version='py3',image_scope='inference', instance_type='ml.p3.16xlarge')

# print(image_uris.)

import json
from inference import model_fn, predict_fn, input_fn, output_fn

payload = json.dumps({
    "text": "What is the capital of Spain?"
})

response, accept = output_fn(
    predict_fn(
        input_fn(payload, "application/json"),
        model_fn("../")
    ),
    "application/json"
)

print(response)