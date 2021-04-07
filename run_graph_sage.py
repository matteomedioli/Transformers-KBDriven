from dgn import GraphSageEmbeddingUnsup
import torch
import time
from datetime import datetime
from utils import cuda_setup, create_data_split, Config
import os
import json

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
assert(config.dgn.embedding_size == config.embedding.hidden_size)

# Init main Model
model = GraphSageEmbeddingUnsup(config)
model = model.to(device)
for layer in model.modules():
    print(layer)

for epoch in range(1, 51):
    loss = model.unsupervised_training(data.x, data.edge_index)
    print(f'Epoch: {epoch:03d}, Total GraphSage Loss: {loss:.4f}, ')

for epoch in range(1, 151):
    loss = model.unsupervised_training(data.x, data.edge_index)
    print(f'Epoch: {epoch:03d}, Total GraphSage Loss: {loss:.4f}, ')
    if epoch in [10,25,50,75,100,125,150]:
        path = "/data/medioli/models/dgn/graphsage/epoch"+str(epoch)+"/"
        os.mkdir(path)
        torch.save(model.state_dict(), path)
        os.popen("cp config.json "+path+"config.json")
        os.popen("cp nohup.out "+path+"log.out")

