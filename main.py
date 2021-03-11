
from dgn import MainModel, WordnetEmbeddings, DGN
import torch
from training_mlm import train_mlm
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
from collections import Counter
from wikidataset import WikiCorpora
import time
import logging
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import cuda_setup, get_batches, Config, insert_between
import os
from transformers import AutoTokenizer
import json
from utils import plot_pca
from tqdm import tqdm


# time_start = time.time()
# date_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
# logging.basicConfig(filename="data/"+str(date_time)+".log", level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# W = WordNetGraph()
# C = WikiCorpora(2, "data/corpora/enwiki.txt")


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = cuda_setup()
""" 
D = torch.load("data/graphs/wordnet_ids.pt", map_location=torch.device('cpu'))
print(np.unique(D.root))
print(np.unique(D.lemma))
print(np.unique(D.pos))
print(np.unique(D.sense))
"""

W = torch.load("/data/medioli/wordnet/wordnet_ids.pt")
W = W.to(device)
print(W)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)

model = MainModel(config)
model = model.to(device)
node_embeddings, names = model(W)
train_mlm("train_transformers.json", node_embeddings)

# layer_info = json.load("config.json")
# dataset = torch.load("data/graphs/node_labels.pt")
# create_text_gnn(layer_info, dataset, device)
#
# from transformers import BertTokenizer, BertModel
# from gcnn import WordnetEmbeddings
# tokenizer = BertTokenizer("bert-base-cased")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# model = WordnetEmbeddings()
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

