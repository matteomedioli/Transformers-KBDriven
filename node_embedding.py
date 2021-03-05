from utils import cuda_setup, get_batches
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import Node2Vec
from torch import argmax
from torch.nn.modules.loss import _Loss
import logging
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def node2vec(graph, target_feature):
    device = cuda_setup()
    model = Node2Vec(graph.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    train_perc = int((len(graph.x) * 80) / 100)
    train_mask = torch.tensor([True if i <= train_perc else False for i, _ in enumerate(graph.x)])
    test_mask = torch.tensor([True if i > train_perc else False for i, _ in enumerate(graph.x)])

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[train_mask], graph.target_feature[train_mask], z[test_mask], graph.target_feature[test_mask],
                         max_iter=150)
        return acc

    loss = train()
    acc = test()
    return model, loss, acc  # Ritorna i Node embedding


class OHELoss(_Loss):
    def __init__(self):
        super(OHELoss, self).__init__()

    def forward(self, ohe_embeddings, target, indices):
        """ loss function called at runtime """
        loss = 0
        for i in indices[:-1]:
            class_loss = F.nll_loss(
                F.log_softmax(ohe_embeddings[:, i:i + 1], dim=1),
                argmax(target[:, i:i + 1])
            )
            loss += class_loss
            return loss


class NodeAutoencoder(nn.Module):
    def __init__(self, input_shape: int = 235233):
        tic = time.perf_counter()
        logging.debug('*** AUTOENCODER ***')
        super().__init__()
        logging.debug('1) Initializing Encoder')
        logging.debug('First Layer')
        self.encode1 = nn.Linear(input_shape, 131072)
        logging.debug('Second Layer')
        self.encode2 = nn.Linear(131072, 65536)
        logging.debug('THird Layer')
        self.encode3 = nn.Linear(65536, 16384)
        logging.debug('Fourth Layer')
        self.encode4 = nn.Linear(16384, 4096)

        logging.debug('2) Initializing Decoder')
        logging.debug('First Layer')
        self.decode1 = nn.Linear(4096, 16384)
        logging.debug('Second Layer')
        self.decode2 = nn.Linear(16384, 65536)
        logging.debug('Third Layer')
        self.decode2 = nn.Linear(65536, 131072)
        logging.debug('Fourth Layer')
        self.decode4 = nn.Linear(131072, input_shape)
        toc = time.perf_counter()
        logging.debug(f"Layers init complete in {toc - tic:0.4f} seconds")
        logging.debug('')

    def encode(self, x: torch.Tensor):
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        x = F.relu(self.encode3(x))
        return x

    def decode(self, x: torch.Tensor):
        x = F.relu(self.decode1(x))
        x = F.relu(self.decode2(x))
        x = F.relu(self.decode3(x))
        return x

    def forward(self, x: torch.Tensor):
        print("AUTOENCODER FORWARD PASS")
        x = self.encode(x)
        x = self.decode(x)
        return x


def train_autoencoder(data):
    net = NodeAutoencoder()
    torch.save("data/graphs/autoencoder.pt")
    logging.debug("Saved Autoencoders in data/graphs/autoencoder.pt")
    optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for epoch in range(250):
        logging.debug('Epoch: ' + epoch)
        for batch in get_batches(data):
            net.zero_grad()
            # Pass batch through
            output = net(batch)
            loss_fn = torch.nn.CosineEmbeddingLoss(reduction='none')
            # Get Loss + Backprop
            loss = loss_fn(output, batch).sum()  #
            # loss = OHELoss()
            losses.append(loss)
            loss.backward()
            optimizer.step()
    logging.debug("DONE!")
    return net, losses
