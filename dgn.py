import torch_geometric
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import cuda_setup
from torch_cluster import random_walk
from torch_geometric.nn import SAGEConv
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from nltk.corpus import wordnet as wn
import numpy as np

device = cuda_setup()


class CustomBERTModel(nn.Module):
    def __init__(self, bert_model, node_dict, tokenizer, compute_node_embeddings=True):
        super(CustomBERTModel, self).__init__()
        self.compute_node_embeddings = compute_node_embeddings
        self.node_dict = node_dict
        self.tokenizer = tokenizer
        self.node = None

        self.bert = bert_model
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 3)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None, 
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        if self.compute_node_embeddings:
            output_hidden_states = True
            node_batch = []
            for batch in input_ids:
                words_node = []
                for ids in batch:
                    # synset_embedding = self.node_embeddings[wn.synsets(self.tokenizer.decode(ids))[0]]
                    # words_node.append(synset_embedding)
                    if wn.lemmas(self.tokenizer.decode(ids)):
                        lemma_embedding = self.node_dict[str(wn.lemmas(self.tokenizer.decode(ids))[0])[7:-2]]
                        words_node.append(torch.tensor(lemma_embedding))
                        # print(self.tokenizer.decode(ids), str(wn.lemmas(self.tokenizer.decode(ids))[0])[7:-2], lemma_embedding)
                    else:
                        words_node.append(torch.zeros(768))
                words_node_t = torch.stack(words_node)
                print("WORDS_NODE_T", words_node_t.shape)
                node_batch.append(words_node_t)
            word_node_embeddings = torch.stack(node_batch)
            print(word_node_embeddings.shape)
        print(output_hidden_states) 
        outputs = self.bert(input_ids,
                            attention_mask, 
                            token_type_ids, 
                            position_ids, 
                            head_mask,
                            inputs_embeds,
                            encoder_hidden_states,
                            labels,
                            output_attentions,
                            output_hidden_states,
                            return_dict)
        # TODO: sa dio perchè torna loss None, chekc su hidden states da fare. Non mollare lesionato 
        print(outputs)
        word_hidden_states = outputs["hidden_states"][0]
        print(word_hidden_states.shape)
        print(word_node_embeddings.shape)
        return outputs


class GraphSageEmbeddingUnsup(torch.nn.Module):
    def __init__(
            self,
            config
    ):
        super(GraphSageEmbeddingUnsup, self).__init__()
        self.config = config
        self.embedding = WordnetEmbeddings(self.config)
        self.dgn = SAGE(self.config)

    def forward(self, x, adjs):
        input_ids = x
        x = self.embedding(input_ids)
        # plot_pca(pyg_graph.x.tolist(), colors=pyg_graph.node_type, n_components=3, element_to_plot=5000)
        node_embeddings = self.dgn(x, adjs)
        # plot_pca(node_embeddings, colors=pyg_graph.node_type, n_components=3, element_to_plot=5000)
        return node_embeddings

    def full_forward(self, x, edge_index, epoch):
        input_ids = x
        x = self.embedding(input_ids)
        # plot_pca(pyg_graph.x.tolist(), colors=pyg_graph.node_type, n_components=3, element_to_plot=5000)
        node_embeddings = self.dgn.full_forward(x, edge_index).cpu().detach().numpy()
        # path = "/data/medioli/models/dgn/graphsage_w10/epoch" + str(epoch) + "/"
        return node_embeddings

    def full_forward(self, x, edge_index, epoch):
        input_ids = x
        x = self.embedding(input_ids)
        # plot_pca(pyg_graph.x.tolist(), colors=pyg_graph.node_type, n_components=3, element_to_plot=5000)
        node_embeddings = self.dgn.full_forward(x, edge_index).cpu().detach().numpy()
        # path = "/data/medioli/models/dgn/graphsage_w10/epoch" + str(epoch) + "/"
        # plot_pca(node_embeddings, colors=None, n_components=2, element_to_plot=150000, path=path)
        return node_embeddings


class WordnetEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.synset_embeddings = nn.Embedding(self.config.embedding.synset_vocab_size,
                                              self.config.embedding.hidden_size)
        self.lemma_embeddings = nn.Embedding(self.config.embedding.lemma_vocab_size, self.config.embedding.hidden_size)
        self.pos_type_embeddings = nn.Embedding(self.config.embedding.pos_types, self.config.embedding.hidden_size)
        self.sense_embeddings = nn.Embedding(self.config.embedding.tot_sense, self.config.embedding.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(self.config.embedding.hidden_size, eps=self.config.embedding.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.embedding.hidden_dropout_prob)

    def forward(self, x):
        synset_embeds = self.synset_embeddings(x[:, 0])
        pos_embeds = self.pos_type_embeddings(x[:, 1])
        sense_embeds = self.sense_embeddings(x[:, 2])
        lemma_embeds = self.lemma_embeddings(x[:, 3])
        embeddings = synset_embeds + lemma_embeds
        embeddings += (pos_embeds + sense_embeds)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=10,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, config):
        super(SAGE, self).__init__()
        self.config = config
        self.num_layers = self.config.sage.num_layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = self.config.embedding.hidden_size if i == 0 else self.config.sage.hidden_channels
            self.convs.append(SAGEConv(in_channels, self.config.sage.hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.config.sage.dropout, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.config.sage.dropout, training=self.training)
        return x


class DGN(nn.Module):
    def __init__(self, config):
        super(DGN, self).__init__()

        self.in_dim = config.embedding.hidden_size
        self.out_dim = config.dgn.embedding_size
        self.type = config.dgn.type

        if self.type == 'gcn':
            builderName = "GCNConv"
        elif self.type == 'gat':
            builderName = "GCNConv"
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))

        targetClass = getattr(torch_geometric.nn, builderName)

        self.layers = nn.ModuleList()
        self.act = F.relu

        if not config.dgn.hidden_sizes_list:
            layer_instance = targetClass(self.in_dim, self.out_dim)
            self.layers.append(layer_instance)
        else:
            layer_instance = targetClass(self.in_dim, config.dgn.hidden_sizes_list[0])
            self.layers.append(layer_instance)
            for i in range(len(config.dgn.hidden_sizes_list) - 1):
                in_dim = config.dgn.hidden_sizes_list[i]
                out_dim = config.dgn.hidden_sizes_list[i + 1]
                layer_instance = targetClass(in_dim, out_dim)
                self.layers.append(layer_instance)

        if config.dgn.batch_norm:
            self.bn = torch.nn.BatchNorm1d(self.out_dim)
        else:
            self.bn = None

        if config.dgn.dropout:
            self.dropout = torch.nn.Dropout()
        else:
            self.dropout = None

    def forward(self, x, pyg_graph, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(device)
        graph_embeddings = torch.zeros(1, 2 * self.out_dim).to(device)  # 2 gap + gmp = 128 + 128
        if self.dropout:
            x = self.dropout(x)
        for l in self.layers:
            x = self.act(l(x, pyg_graph.edge_index))
            x_i = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            graph_embeddings += x_i
        if self.bn:
            x = self.bn(x)
        node_embeddings = x
        print(node_embeddings.shape, graph_embeddings.shape)
        return node_embeddings.detach(), graph_embeddings.detach()
