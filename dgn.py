import torch_geometric
from torch_geometric.nn import GraphConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class MainModel(torch.nn.Module):
    def __init__(
            self,
            config
    ):
        super(MainModel, self).__init__()

        self.input_embedding_size = config.embedding.hidden_size
        self.node_embedding_size = config.dgn.embedding_size

        self.embedding = WordnetEmbeddings(config)
        self.dgn = DGN(config)

    def forward(self, pyg_graph):
        input_ids = pyg_graph.x
        pyg_graph.x = self.embedding(input_ids)
        node_embeddings = self.dgn(pyg_graph)
        print(len(node_embeddings))


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
                pre = config.dgn.hidden_sizes_list[i]
                post = config.dgn.hidden_sizes_list[i + 1]
                layer_instance = targetClass(pre, post)
                self.layers.append(layer_instance)

        if config.dgn.batch_norm:
            self.bn = torch.nn.BatchNorm1d(self.out_dim)

        if config.dgn.dropout:
            self.dropout = torch.nn.Dropout()

    def forward(self, x, pyg_graph, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long()
        graph_embeddings = None
        if self.dropout:
            x = self.dropout(x)
        for l in self.layers:
            x = self.act(l(x, pyg_graph.edge_index))
            x_i = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            graph_embeddings += x_i
        if self.bn:
            x = self.bn(x)
        node_embeddings = x

        return node_embeddings.detach(), graph_embeddings.detach()


class WordnetEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.synset_embeddings = nn.Embedding(config.embedding.lemma_vocab_size, config.embedding.hidden_size)
        self.lemma_embeddings = nn.Embedding(config.embedding.synset_vocab_size, config.embedding.hidden_size)
        self.pos_type_embeddings = nn.Embedding(config.embedding.pos_types, config.embedding.hidden_size)
        self.sense_embeddings = nn.Embedding(config.embedding.tot_sense, config.embedding.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding.hidden_size, eps=config.embedding.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding.hidden_dropout_prob)

    def forward(self, x):
        import numpy as np
        print(len(np.unique(x[:, 0].tolist())))
        print(len(np.unique(x[:, 1].tolist())))
        print(len(np.unique(x[:, 2].tolist())))
        print(len(np.unique(x[:, 3].tolist())))

        synset_embeds = self.synset_embeddings(x[:, 0])
        pos_embeds = self.pos_type_embeddings(x[:, 1])
        sense_embeds = self.sense_embeddings(x[:, 2])
        lemma_embeds = self.lemma_embeddings(x[:, 3])
        embeddings = synset_embeds + lemma_embeds
        embeddings += (pos_embeds + sense_embeds)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
