import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, TopKPooling, global_mean_pool as gap, \
    global_max_pool as gmp
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import cuda_setup, plot_pca
from torch_cluster import random_walk
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from nltk.corpus import wordnet as wn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch import Tensor
import random
device = cuda_setup()


def weight_init(m):
    m.reset_parameters()


class BertForWordNodeRegression(nn.Module):
    def __init__(self, node_dict, tokenizer, bert_model, regression_model, reg_lambda=1, graph_regularization=True):
        super(BertForWordNodeRegression, self).__init__()
        self.graph_regularization = graph_regularization
        self.node_dict = node_dict
        self.tokenizer = tokenizer
        self.reg_lambda = reg_lambda

        self.bert = bert_model

        self.regression = regression_model

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
                return_dict=None,
                regression_criterion=nn.MSELoss()):
        if self.graph_regularization:

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
                        # print(self.tokenizer.decode(ids), str(wn.lemmas(self.tokenizer.decode(ids))[0])[7:-2],
                        # lemma_embedding)
                    else:
                        words_node.append(torch.full([64], fill_value=1, dtype=torch.float))
                words_node_t = torch.stack(words_node)
                node_batch.append(words_node_t)
            word_node_embeddings = torch.stack(node_batch)

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            labels=labels,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
        if self.graph_regularization:
            word_hidden_states = outputs["hidden_states"][0]
            regression_valid_idx = []

            for nodes_text_tensor in word_node_embeddings:
                idx_word_with_node = [i for i, lemma_embedding in enumerate(nodes_text_tensor) if
                                      not torch.eq(torch.sum(lemma_embedding), 64)]
                regression_valid_idx.append(idx_word_with_node)

            regression_out = self.regression(word_hidden_states)
            print(regression_out.shape, word_node_embeddings.shape)
            for i, r, n in enumerate(zip(regression_out, word_node_embeddings)):
                print(i, r.shape, n.shape)

            regression_loss = regression_criterion(regression_out, word_node_embeddings.to(device))
            print("REG LOSS: ", regression_loss)
            outputs["loss"] = outputs["loss"] + (self.reg_lambda * regression_loss)

        return outputs


class WordnetDGN(torch.nn.Module):
    def __init__(
            self,
            config,
            model_path
    ):
        super(WordnetDGN, self).__init__()
        self.config = config
        self.embedding = WordnetEmbeddings(self.config)
        self.dgn = ConvDGN(self.config)
        self.model_path = model_path
        self.reset_parameters()

    def forward(self, x, adjs, edge_attrs=None):
        assert edge_attrs is not None and self.config.dgn.type == "rgcn"
        input_ids = x
        x = self.embedding(input_ids)
        if self.config.dgn.type == "rgcn":
            node_embeddings = self.dgn(x, adjs, edge_attrs)
        else:
            node_embeddings = self.dgn(x, adjs)
        return node_embeddings

    def full_forward(self, x, edge_index, node_root_colors, epoch, edge_attrs=None):
        assert edge_attrs is not None and self.config.dgn.type == "rgcn"
        input_ids = x
        x = self.embedding(input_ids)
        if self.config.dgn.type == "rgcn":
            node_embeddings = self.dgn.full_forward(x, edge_index, edge_attrs).cpu().detach().numpy()
        else:
            node_embeddings = self.dgn.full_forward(x, edge_index).cpu().detach().numpy()
        if epoch > 0:
            plot_pca(node_embeddings, node_root_colors, n_components=3, element_to_plot=300000, path=self.model_path,
                     epoch=epoch)
        return node_embeddings

    def reset_parameters(self):
        self.dgn.reset_parameters()


class Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Regression, self).__init__()
        self.num_layers = num_layers
        # Iterable nn.Layers list
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size if i != self.num_layers - 1 else output_size
            # Linear Layer
            self.layers.append(nn.Linear(in_channels, out_channels))
            # Activation
            if i != self.num_layers - 1:
                self.layers.append(nn.LeakyReLU())
                # Dropout if you want
                # self.layers.append(nn.Dropout())
            else:
                self.layers.append(nn.Softmax())

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def info(self):
        for layer in self.modules():
            print(layer)


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
        self.dropout = F.dropout

    def forward(self, x):
        synset_embeds = self.synset_embeddings(x[:, 0])
        pos_embeds = self.pos_type_embeddings(x[:, 1])
        sense_embeds = self.sense_embeddings(x[:, 2])
        lemma_embeds = self.lemma_embeddings(x[:, 3])
        embeddings = synset_embeds + lemma_embeds
        embeddings += (pos_embeds + sense_embeds)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=self.training)
        return embeddings


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=2,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class ConvDGN(nn.Module):
    def __init__(self, config):
        super(ConvDGN, self).__init__()

        self.in_dim = config.embedding.hidden_size
        self.out_dim = config.dgn.embedding_size
        self.type = config.dgn.type

        if self.type == "gcn":
            builderName = "GCNConv"
        elif self.type == "gat":
            builderName = "GATConv"
        elif self.type == "sage":
            builderName = "SAGEConv"
        elif self.type == "rgcn":
            builderName = "RGCNConv"
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))

        targetClass = getattr(torch_geometric.nn, builderName)

        self.conv_layers = nn.ModuleList()
        self.dropout = F.dropout
        self.num_conv = 0

        if not config.dgn.hidden_sizes_list:
            if self.type == "rgcn":
                conv_layer_instance = targetClass(self.in_dim, self.out_dim, 20, 10)
            else:
                conv_layer_instance = targetClass(self.in_dim, self.out_dim)
            self.conv_layers.append(conv_layer_instance)
            self.num_conv += 1
        else:
            if self.type == "rgcn":
                conv_layer_instance = targetClass(self.in_dim, config.dgn.hidden_sizes_list[0], 20, 10)
            else:
                conv_layer_instance = targetClass(self.in_dim, config.dgn.hidden_sizes_list[0])
            self.conv_layers.append(conv_layer_instance)
            for i in range(len(config.dgn.hidden_sizes_list) - 1):
                in_dim = config.dgn.hidden_sizes_list[i]
                out_dim = config.dgn.hidden_sizes_list[i + 1]
                if self.type == "rgcn":
                    conv_layer_instance = targetClass(in_dim, out_dim, 20, 10)
                else:
                    conv_layer_instance = targetClass(in_dim, out_dim)
                self.conv_layers.append(conv_layer_instance)
                self.num_conv += 1

    def forward(self, x, adjs, edge_attrs=None):
        assert edge_attrs is not None and self.type == "rgcn"
        for i, (edge_index, e, size) in enumerate(adjs):
            rand_edge_attrs = torch.Tensor([random.randint(0,20) for _ in range(e.shape[0])])
            x_target = x[:size[1]]  # Target nodes are always placed first.
            if self.type == "rgcn":
                x = self.conv_layers[i]((x, x_target), edge_index, edge_attrs[:e.shape[0]])
            else:
                x = self.conv_layers[i]((x, x_target), edge_index, edge_attrs)
            if i != self.num_conv - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index, edge_attrs=None):
        assert edge_attrs is not None and self.type == "rgcn"
        for i, conv in enumerate(self.conv_layers):
            if self.type == "rgcn":
                x = conv(x, edge_index, edge_attrs)
            else:
                x = conv(x, edge_index)
            if i != self.num_conv - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
