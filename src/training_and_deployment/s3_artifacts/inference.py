import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from previous_chapters import (
    GPTModel, 
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text
)
import tiktoken


# loads the trained model
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs["gpt2-medium (355M)"])

    model = GPTModel(BASE_CONFIG)

    model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

    model.to(device).eval()

    print("Done loading model.")
    return model


# takes request data and formats it into a form suitable for making predictions
def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(serialized_input_data)
        print(input_data)
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


# takes the formatted request data and performs inference against the loaded model
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    print(input_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Process input_data, convert to tensor, etc.
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_data['text'], tokenizer).to(device),
        max_new_tokens=35,
        context_size=1024,
        eos_id=50256,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    print(generated_text)

    return generated_text


# takes the prediction result and formats it into a response message
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == 'application/json':
        return json.dumps(prediction_output), 'application/json'
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)