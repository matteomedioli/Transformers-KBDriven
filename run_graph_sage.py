from torch.optim import lr_scheduler

from dgn import WordNodeEmbedding, NeighborSampler
import torch.nn.functional as F
import torch
import time
from datetime import datetime
from utils import cuda_setup, create_data_split, Config
import os
import json


def train(model, x, loader, scheduler, epoch):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(x[n_id], adjs, epoch)
            out_split = out.split(out.size(0) // 3, dim=0)
            out = out_split[0]
            pos_out = out_split[1]
            neg_out = out_split[2]

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            optimizer.step()

        total_loss += float(loss) * out.size(0)
    scheduler.step()
    return total_loss / data.num_nodes


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = cuda_setup()

data = torch.load("/data/medioli/wordnet/wordnet_v1.pt")
data = data.to(device)
print(data)

# Load configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
config = Config(config)
assert (config.dgn.embedding_size == config.embedding.hidden_size)

dgn_path = "/data/medioli/models/dgn/graphsage/"
# Init main Model
DGN = WordNodeEmbedding(config, dgn_path)
DGN = DGN.to(device)
for layer in DGN.modules():
    print(layer)

train_loader = NeighborSampler(data.edge_index, sizes=config.dgn.sizes, batch_size=config.dgn.batch_size,
                               shuffle=True, num_nodes=data.num_nodes)

optimizer = torch.optim.Adam(DGN.parameters(), lr=config.dgn.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

for epoch in range(1, 151):
    train_loss = train(DGN, data.x, train_loader, scheduler, epoch)
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}')
    if epoch in [1, 10, 25, 50, 75, 100, 125, 150]:
        path = dgn_path
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(DGN.state_dict(), path+str(epoch)+"e_model.pt")
        DGN.eval()
        DGN.full_forward(data.x, data.edge_index, epoch)
