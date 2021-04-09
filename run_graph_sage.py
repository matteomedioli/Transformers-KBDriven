from dgn import GraphSageEmbeddingUnsup, NeighborSampler
import torch.nn.functional as F
import torch
import time
from datetime import datetime
from utils import cuda_setup, create_data_split, Config
import os
import json


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


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = cuda_setup()

data = torch.load("/data/medioli/wordnet/wordnet_ids.pt")
data = create_data_split(data)
data = data.to(device)
print(data)

# Load configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)
assert (config.dgn.embedding_size == config.embedding.hidden_size)

# Init main Model
GS = GraphSageEmbeddingUnsup(config)
GS = GS.to(device)
for layer in GS.modules():
    print(layer)

train_loader = NeighborSampler(data.edge_index, sizes=config.dgn.sizes, batch_size=config.dgn.batch_size,
                               shuffle=True, num_nodes=data.num_nodes)
optimizer = torch.optim.Adam(GS.parameters(), lr=config.dgn.learning_rate)

for epoch in range(1, 151):
    train_loss = train(GS, data.x, train_loader)
    print(f'Epoch: {epoch:03d}, Total GraphSage Loss: {train_loss:.4f}, ')
    if epoch in [1, 10, 25, 50, 75, 100, 125, 150]:
        path = "/data/medioli/models/dgn/graphsage/epoch" + str(epoch) + "/"
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
            print(path)
        torch.save(GS.state_dict(), path + "model.pt")
