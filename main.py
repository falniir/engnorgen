import os
import yaml
import logging
from datasets import load_dataset
from ludwig.api import LudwigModel
import pandas as pd


# Load the dataset
dataset = load_dataset("NbAiLab/norwegian-alpaca", cache_dir="/tmp/Norgen")


# Convert the dataset to a pandas DataFrame
dataset_df = dataset['train'].to_pandas()





config_str = """
model_type: llm
base_model: bineric/NorskGPT-Mistral-7b
quantization:
  bits: 4
adapter:
  type: lora
prompt:
  template: |
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:

input_features:
  - name: combined_input
    type: text
    preprocessing:
      max_sequence_length: 512  # Adjusted to allow longer texts

output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 256

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1  # Adjust as needed based on your dataset size and training needs
"""
config = yaml.safe_load(config_str)

# Preprocess dataset to create a combined input
def combine_inputs(example):
    # Replace None with empty string before concatenation
    instruction = example['instruction'] if example['instruction'] is not None else ""
    input_text = example['input'] if example['input'] is not None else ""
    example['combined_input'] = f"### Instruction:\n{instruction}\n### Input:\n{input_text}"
    return example

dataset = dataset.map(combine_inputs)

# Initialize and train the model
model = LudwigModel(config=config, logging_level=logging.INFO)

# Now train the model using this DataFrame
results = model.train(dataset_df)  # Ensure the correct subset of the dataset is used

# Save model
model.save("model")
