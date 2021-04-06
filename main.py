from dgn import MainModel, NeighborSampler
import torch
import time
import logging
from datetime import datetime
from utils import cuda_setup, get_batches, create_data_split, Config, insert_between
import os
import json
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T

time_start = time.time()
date_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
# logging.basicConfig(filename="data/" + str(date_time) + ".log", level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = cuda_setup()


def train(model, x, loader):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test(model, x, edge_index):
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask].cpu())

    val_accuracy = clf.score(out[data.val_mask], data.y[data.val_mask].cpu())
    test_accuracy = clf.score(out[data.test_mask], data.y[data.test_mask].cpu())

    return val_accuracy, test_accuracy


data = torch.load("/data/medioli/wordnet/wordnet_ids.pt")
data = create_data_split(data)
data.y = torch.Tensor(data.node_type)
data = data.to(device)
print(data)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)

model = MainModel(config)
model = model.to(device)
for layer in model.modules():
    print(layer)

train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 51):
    loss = train(model, data.x, train_loader)
    val_acc, test_acc = test(model, data.x, data.edge_index)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')


# TODO: Add integration
#  for https://github.com/stellargraph/stellargraph/blob/ec132647e5cf43ff683e3f2e72e18ac6daa98202/stellargraph/layer/hinsage.py
