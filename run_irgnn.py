import json
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from dgn import NNConv, WordnetEmbeddings
from auto_encoder import AutoEncoder
from dgn import NeighborSampler
from utils import cuda_setup, create_data_split, Config

# model parameters
device = cuda_setup()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_epoch = 300
torch.manual_seed(1234)
node_hidden_dim = edge_hidden_dim = 64
num_step_message_passing = 2
lr = 1e-3

val_ratio = 0.1
test_ratio = 0.1

edge_type = 21
batch_size = 512

# load data
data = torch.load("/data/medioli/wordnet/wordnet_v1.pt")
print(data)

class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.wordnet_embedding = WordnetEmbeddings(config)
        self.lin0 = torch.nn.Linear(config.embedding.hidden_size, node_hidden_dim)
        self.lin_h = torch.nn.Linear(node_hidden_dim, node_hidden_dim)
        self.lin_h_m = torch.nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.edge_network = Sequential(Linear(edge_type + node_hidden_dim, edge_hidden_dim), ReLU(),
                                       Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, self.edge_network, aggr='mean', root_weight=False)
        self.gru = GRU(node_hidden_dim, node_hidden_dim)

    def forward(self, x, edge_index, edge_attr):

        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(num_step_message_passing):
            prev = h
            m = F.relu(self.conv(out, edge_index, edge_attr))
            h = F.relu(self.lin_h(h))
            out = F.relu(self.lin_h_m(torch.cat((h.squeeze(0), m), dim=1))) + prev
            out = out.squeeze(0)
        out = F.relu(out)

        return out


# Load configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)
assert (config.dgn.embedding_size == config.embedding.hidden_size)

model = AutoEncoder(node_hidden_dim, Encoder(config)).to(device)

data.train_mask = data.val_mask = data.test_mask = None
tr_data, val_data, ts_data = model.split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)
tr_loader = NeighborSampler(tr_data, size=[5] * num_step_message_passing,
                            num_hops=num_step_message_passing, batch_size=batch_size, bipartite=False, shuffle=True)
val_loader = NeighborSampler(val_data, size=[5] * num_step_message_passing,
                             num_hops=num_step_message_passing, batch_size=batch_size, bipartite=False)
ts_loader = NeighborSampler(ts_data, size=[5] * num_step_message_passing,
                            num_hops=num_step_message_passing, batch_size=batch_size, bipartite=False)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    l = []
    tr_x, tr_edge_index, tr_edge_attr = tr_data.x.to(
        device), tr_data.edge_index.to(device), tr_data.edge_attr.to(device)
    model.train()

    for sub_data in tr_loader():
        sub_data = sub_data.to(device)
        optimizer.zero_grad()
        z = tr_x.new_zeros(tr_x.size(0), node_hidden_dim)
        z[sub_data.n_id] = model.encode(tr_x[sub_data.n_id], sub_data.edge_index, tr_edge_attr[sub_data.e_id])
        loss = model.log_loss(z, tr_edge_index[:, sub_data.e_id], tr_edge_attr[sub_data.e_id])
        l.append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return sum(l) / len(l)


def test():
    model.eval()
    with torch.no_grad():
        ts_x, ts_edge_index, ts_edge_attr = ts_data.x.to(
            device), ts_data.edge_index.to(device), ts_data.edge_attr.to(device)

        # load learned embeddings
        tr_x, tr_edge_index, tr_edge_attr = tr_data.x.to(device), tr_data.edge_index.to(device), tr_data.edge_attr.to(
            device)
        z = model.encode(tr_x, tr_edge_index, tr_edge_attr)

    return model.test(z, ts_edge_index, ts_edge_attr)


for epoch in range(n_epoch):
    tr_loss = train()
    auc, precision, recall, acc = test()
    print('Epoch: {:03d}, tr-loss: {:.6f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, ACC: {:.4f}'.format(epoch,
                                                                                                               tr_loss,
                                                                                                               auc,
                                                                                                               precision,
                                                                                                               recall,
                                                                                                               acc))
