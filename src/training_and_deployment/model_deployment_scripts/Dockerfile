# Use the SageMaker PyTorch image as the base image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-gpu-py310

# Install the additional dependencies
RUN pip install "numpy"
RUN pip install "torch"
RUN pip install "tiktoken"
RUN pip install "tqdm"