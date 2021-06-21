from transformers import AutoTokenizer

from utils import load_model, Config
from probing import *
import json
import torch

model_reg = load_model("/home/med/Scrivania/models/rgcn/checkpoint-740000/", "pytorch_model.bin", True, "cpu")
model = load_model("/home/med/Scrivania/models/baseline/checkpoint-740000/", "pytorch_model.bin", False, "cpu")

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=30,
        return_special_tokens_mask=True,
    )

def bert_batcher(params, batch):
    examples = [' '.join(s) for s in batch]
    tokenized_batch = tokenize_function(examples)
    return model(torch.tensor(tokenized_batch.input_ids))

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)
probe_len = LengthEval("/home/med/Scrivania/data/probing")
probe_len.run(config, bert_batcher)
