import os
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
import argparse
import probing
from transformers import BertForMaskedLM
from utils import load_custom_model, Config
import json
import torch

tasks = {
    1:"LengthEval",
    2:"WordContentEval",
    3:"DepthEval",
    4:"TopConstituentsEval",
    5:"BigramShiftEval",
    6:"TenseEval",
    7:"SubjNumberEval",
    8:"ObjNumberEval",
    9:"OddManOutEval",
    10:"CoordinationInversionEval"
}

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=30,
        return_special_tokens_mask=True,
    )


def bert_batcher(model, batch, add_node_embedding):
    examples = []
    node_batch = []
    for sentence in batch:
        examples.append(' '.join(sentence))
        if add_node_embedding:
            if os.path.exists("/data/medioli/wordnet/node_dict_w2_rgcn_e150.pt"):
                node_dict = torch.load("/data/medioli/wordnet/node_dict_w2_rgcn_e150.pt")
            words_node = []
            for s in sentence:
                if wn.lemmas():
                    lemma_embedding = node_dict[str(wn.lemmas(s)[0])[7:-2]]
                    print(lemma_embedding)
                    print(str(wn.lemmas(s)[0])[7:-2])
                    print(node_dict[str(wn.lemmas(s)[0])[7:-2]])
                    words_node.append(torch.tensor(lemma_embedding))
                else:
                    words_node.append(torch.full([64], fill_value=1, dtype=torch.float))
            words_node_t = torch.stack(words_node)
            node_batch.append(words_node_t)
        #word_node_embeddings = torch.stack(node_batch)

    tokenized_batch = tokenize_function(examples)
    with torch.no_grad():
        out = model(torch.tensor(tokenized_batch.input_ids), output_hidden_states=True)
        last_hidden_states = out["hidden_states"][0]
    return last_hidden_states[:,0,:]

# python3 evaluate_model.py -m custom_base -p 1

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--probing", required=True,
                help="Probing Task index\n"+ str(tasks))
ap.add_argument("-m", "--model", required=True,
                help="Model to evaluate: pretrained, ext_pretrained, custom_base, custom_reg")
args = vars(ap.parse_args())

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)
print(config)
print(config.paths[config.env])

models_folder = config.paths[config.env].models
tasks_folder = config.paths[config.env].tasks

add_node_embedding = False

if args["model"] == "pretrained":
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
elif args["model"] == "ext_pretrained":
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    add_node_embedding = True
elif args["model"] == "custom_base":
    model = load_custom_model(models_folder + "baseline/", "pytorch_model.bin", False, "cpu")
elif args["model"] == "custom_reg":
    model = load_custom_model(models_folder+"rgcn/", "pytorch_model.bin", True, "cpu")
else:
    print("Choose a correct model to test: pretrained, ext_pretrained, custom_base, custom_reg")
    exit(0)

task_index = args["probing"]

task = getattr(probing, tasks[int(task_index)])
probe = task(tasks_folder)
results = probe.run(model, config, bert_batcher, add_node_embedding)
print("RESULTS\n", results, "\n")
