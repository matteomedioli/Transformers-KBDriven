import torch
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


class WordNodeRegression:
    def __init__(self, node_embeddings, tokenizer):
        self.tokenizer = tokenizer
        self.node_embeddings = node_embeddings

    def compute_batch_loss(self, input_ids, last_hidden_states):
        node_batch = []
        for batch in self.input_ids:
            words_node = []
            for ids in batch:
                # synset_embedding = self.node_embeddings[wn.synsets(self.tokenizer.decode(ids))[0]]
                # words_node.append(synset_embedding)
                lemma_embedding = self.node_embeddings[wn.lemmas(self.tokenizer.decode(ids))[0]]
                words_node.append(lemma_embedding)
            node_batch.append(words_node)
        tensor_batch = torch.tensor(node_batch)
