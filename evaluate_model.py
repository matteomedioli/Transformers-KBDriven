from transformers import AutoTokenizer
from utils import load_model, Config
from probing import *
import json
import torch


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=30,
        return_special_tokens_mask=True,
    )


def bert_batcher(model, batch):
    examples = [' '.join(s) for s in batch]
    tokenized_batch = tokenize_function(examples)
    with torch.no_grad():
        out = model(torch.tensor(tokenized_batch.input_ids))
        hidden_states = out["last_hidden_state"]
    return hidden_states


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)

model = load_model("/home/med/Scrivania/models/baseline/checkpoint-740000/", "pytorch_model.bin", False, "cpu")
model_reg = load_model("/home/med/Scrivania/models/rgcn/checkpoint-740000/", "pytorch_model.bin", True, "cpu")

probe = ObjNumberEval("/home/med/Scrivania/data/probing")
baseline = probe.run(model, config, bert_batcher)
regular = probe.run(model_reg, config, bert_batcher)
print("BASELINE\n", baseline, "\n")
print("RGCN\n", regular, "\n")


probe = DepthEval("/home/med/Scrivania/data/probing")
baseline = probe.run(model, config, bert_batcher)
regular = probe.run(model_reg, config, bert_batcher)
print("BASELINE\n", baseline, "\n")
print("RGCN\n", regular, "\n")


probe = CoordinationInversionEval("/home/med/Scrivania/data/probing")
baseline = probe.run(model, config, bert_batcher)
regular = probe.run(model_reg, config, bert_batcher)
print("BASELINE\n", baseline, "\n")
print("RGCN\n", regular, "\n")

 
probe = SubjNumberEval("/home/med/Scrivania/data/probing")
baseline = probe.run(model, config, bert_batcher)
regular = probe.run(model_reg, config, bert_batcher)
print("BASELINE\n", baseline, "\n")
print("RGCN\n", regular, "\n")


probe = OddManOutEval("/home/med/Scrivania/data/probing")
baseline = probe.run(model, config, bert_batcher)
regular = probe.run(model_reg, config, bert_batcher)
print("BASELINE\n", baseline, "\n")
print("RGCN\n", regular, "\n")
