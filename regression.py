import torch
from dgn import GraphSageEmbeddingUnsup
from utils import cuda_setup, create_data_split, Config
import json
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

class WordNodeRegression:
    def __init__(self, input_ids, last_hidden_states, tokenizer):
        self.wordnet = torch.load("/data/medioli/wordnet/wordnet_ids.pt")
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.last_hidden_states = last_hidden_states

        # Load configuration file
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        config = Config(config)
        assert (config.dgn.embedding_size == config.embedding.hidden_size)
        model = GraphSageEmbeddingUnsup(config)
        model.load_state_dict(torch.load("/data/medioli/models/dgn/graphsage_w10/epoch50/model.pt"))
        model.eval()

        # self.node_embeddings = model.full_forward(self.wordnet.x, self.wordnet.edge_index)
        for batch in self.input_ids:
            print(batch)
            for ids in batch:
                print(ids, self.tokenizer.decode(ids), wn.synsets(self.tokenizer.decode(ids)))


    def inner_loss(self):
        input_strings = [[self.tokenizer.decode(id_t)] for text in self.input_ids for id_t in text]
        print(input_strings)


