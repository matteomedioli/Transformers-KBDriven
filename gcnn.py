from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GCNN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output,
            dropout=0
    ):
        super(GCNN, self).__init__()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.conv1 = GraphConv(num_input, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv3 = GraphConv(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(num_hidden * 2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

        self.p = dropout

    def forward(self, x, edge_index, batch=None):
        print("START FORWARD")
        if batch is None:
            batch = torch.zeros(x.shape[0]).long()

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        node_embs = x

        x = x1 + x2 + x3

        graph_emb = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x, (node_embs.detach(), graph_emb.detach())


class TextGNN(nn.Module):
    def __init__(self, pred_type, node_embd_type, num_layers, layer_dim_list, act, bn, num_labels, class_weights,
                 dropout):
        super(TextGNN, self).__init__()
        self.node_embd_type = node_embd_type
        self.layer_dim_list = layer_dim_list
        self.num_layers = num_layers
        self.dropout = dropout
        if pred_type == 'softmax':
            assert layer_dim_list[-1] == num_labels
        elif pred_type == 'mlp':
            dims = self._calc_mlp_dims(layer_dim_list[-1], num_labels)
            self.mlp = MLP(layer_dim_list[-1], num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
        self.pred_type = pred_type
        assert len(layer_dim_list) == (num_layers + 1)
        self.act = act
        self.bn = bn
        self.layers = self._create_node_embd_layers()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pyg_graph, dataset):
        acts = [pyg_graph.x]
        for i, layer in enumerate(self.layers):
            ins = acts[-1]
            outs = layer(ins, pyg_graph)
            acts.append(outs)

        return self._loss(acts[-1], dataset)

    def _loss(self, ins, dataset):
        pred_inds = dataset.node_ids
        if self.pred_type == 'softmax':
            y_preds = ins[pred_inds]
        elif self.pred_type == 'mlp':
            y_preds = self.mlp(ins[pred_inds])
        else:
            raise NotImplementedError
        y_true = torch.tensor(dataset.label_inds[pred_inds], dtype=torch.long, device=FLAGS.device)
        loss = self.loss(y_preds, y_true)
        return loss, y_preds.cpu().detach().numpy()

    def _create_node_embd_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            act = self.act if i < self.num_layers - 1 else 'identity'
            layers.append(NodeEmbedding(
                type=self.node_embd_type,
                in_dim=self.layer_dim_list[i],
                out_dim=self.layer_dim_list[i + 1],
                act=act,
                bn=self.bn,
                dropout=self.dropout if i != 0 else False
            ))
        return layers

    def _calc_mlp_dims(self, mlp_dim, output_dim=1):
        dim = mlp_dim
        dims = []
        while dim > output_dim:
            dim = dim // 2
            dims.append(dim)
        dims = dims[:-1]
        return dims

    def forward(self, pyg_graph):
        acts = [pyg_graph.x]
        for i, layer in enumerate(self.layers):
            ins = acts[-1]
            outs = layer(ins, pyg_graph)
            acts.append(outs)
        return acts[-1]


class NodeEmbedding(nn.Module):
    def __init__(self, type, in_dim, out_dim, act, bn, dropout):
        super(NodeEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        if type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'gat':
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        if dropout:
            self.dropout = torch.nn.Dropout()

    def forward(self, ins, pyg_graph):
        if self.dropout:
            ins = self.dropout(ins)
        if self.type == 'gcn':
            if FLAGS.use_edge_weights:
                x = self.conv(ins, pyg_graph.edge_index, edge_weight=pyg_graph.edge_attr)
            else:
                x = self.conv(ins, pyg_graph.edge_index)
        else:
            x = self.conv(ins, pyg_graph.edge_index)
        x = self.act(x)
        return x


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                            self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        else:
            x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class WordnetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
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

    def forward(
            self, synset_id=None, lemma_id=None, pos=None, sense=None, past_key_values_length=0
    ):
        synset_embeds = self.synset_embeddings(synset_id)
        lemma_embeds = self.lemma_embeddings(lemma_id)
        embeddings = synset_embeds + lemma_embeds
        pos_embeds = self.pos_type_embeddings(pos)
        sense_embeds = self.sense_embeddings(sense)
        embeddings += (pos_embeds+sense_embeds)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


''' MODEL LAYER STRUCTURE (code)
        for i in range(self.num_layers):
            act = self.act if i < self.num_layers - 1 else 'identity'
            layers.append(NodeEmbedding(
                type=self.node_embd_type,
                in_dim=self.layer_dim_list[i],
                out_dim=self.layer_dim_list[i + 1],
                act=act,
                bn=self.bn,
                dropout=self.dropout if i != 0 else False
            ))
'''


def create_text_gnn(dataset, device):
    n = '--model'
    pred_type = 'softmax'
    node_embd_type = 'gcn'
    layer_dim_list = [200, num_labels]
    num_layers = len(layer_dim_list)
    class_weights = True
    dropout = True
    s = 'TextGNN:pred_type={},node_embd_type={},num_layers={},layer_dim_list={},act={},' \
        'dropout={},class_weights={}'.format(
        pred_type, node_embd_type, num_layers, "_".join([str(i) for i in layer_dim_list]), 'relu', dropout,
        class_weights
    )

    layer_info = {
        'pred_type': pred_type,
        'node_embd': node_embd_type,
        'layer_dims': layer_dim_list,
        'class_weights': class_weights,
        'dropout': dropout
    }
    lyr_dims = layer_info["layer_dim_list"]
    lyr_dims = [dataset.node_feats.shape[1]] + lyr_dims
    weights = None
    if layer_info["class_weights"].lower() == "true":
        counts = Counter(dataset.label_inds[dataset.node_ids])
        weights = len(counts) * [0]
        min_weight = min(counts.values())
        for k, v in counts.items():
            weights[k] = min_weight / float(v)
        weights = torch.tensor(weights, device=device)

    return TextGNN(
        pred_type=layer_info["pred_type"],
        node_embd_type=layer_info["node_embd_type"],
        num_layers=int(layer_info["num_layers"]),
        layer_dim_list=lyr_dims,
        act=layer_info["act"],
        bn=False,
        num_labels=len(dataset.label_dict),
        class_weights=weights,
        dropout=layer_info["dropout"]
    )
