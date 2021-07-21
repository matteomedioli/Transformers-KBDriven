from __future__ import absolute_import, division, unicode_literals
import random
import torch_geometric
from nltk.corpus import wordnet as wn
from torch_cluster import random_walk
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from transformers import PretrainedConfig
from utils import cuda_setup, plot_pca, get_optimizer
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F

device = cuda_setup()


def weight_init(m):
    m.reset_parameters()


class BertConfigCustom(PretrainedConfig):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


class RobertaConfigCustom(BertConfigCustom):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


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
                idx_word_with_node = [True if not torch.eq(torch.sum(lemma_embedding), 64) else False for
                                      i, lemma_embedding in enumerate(nodes_text_tensor)]
                regression_valid_idx.append(idx_word_with_node)
            regression_valid_idx_mask = torch.tensor(regression_valid_idx)
            regression_out = self.regression(word_hidden_states)
            # print(regression_valid_idx_mask.shape)
            # print(regression_out.shape, word_node_embeddings.shape)
            # print(regression_out[regression_valid_idx_mask].shape, word_node_embeddings[regression_valid_idx_mask].shape)
            # for i, (r, n) in enumerate(zip(regression_out, word_node_embeddings)):
            #     print(i, r.shape, n.shape)
            regression_loss = regression_criterion(regression_out[regression_valid_idx_mask],
                                                   word_node_embeddings[regression_valid_idx_mask].to(device))
            # print("REG LOSS: ", regression_loss)
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
            rand_edge_attrs = torch.Tensor([random.randint(0, 20) for _ in range(e.shape[0])])
            x_target = x[:size[1]]  # Target nodes are always placed first.
            if self.type == "rgcn":
                x = self.conv_layers[i]((x, x_target), edge_index, rand_edge_attrs)  # edge_attrs[:e.shape[0]])
            else:
                x = self.conv_layers[i]((x, x_target), edge_index)
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


class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)):]
            devidx = permutation[0:int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')

        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
            accuracy = 1.0 * correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas


class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg
