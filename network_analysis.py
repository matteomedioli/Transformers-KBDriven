# coding=utf-8
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric
import pandas as pd
d = torch.load("/home/med/Scrivania/data/synsets.pt")
print(d.x[:10])
print(d.name[:10])

#g = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr='Edge_label', create_using=nx.DiGraph(),)
'''
# --------------------------------------------------------
# this graph works without timestamps
# deemed as a final network on the last timestamp
for link in raws:
    node_in = int(link[0])
    node_out = int(link[1])
    if not g.has_edge(node_in, node_out):
        g.add_edge(node_in, node_out)
    if not nxG.has_edge(node_in, node_out):
        nxG.add_edge(node_in, node_out)

# --------------------------------------------------------

# ******
# Q 1 : N => 167,L => 3250,p => 0.23,ED => 39,VarD => 993
# ******
N = len(g.vertices())
L = len(g.edges())
p = nx.density(nxG)
degree = []
for node in nx.degree(nxG):
    degree.append(node[1])
ED = np.sum(degree) / N
VarD = np.var(degree)

# ******
# Q 2 : plot
# ******
nx.draw_networkx(nxG,pos=nx.random_layout(nxG),alpha=0.9,width=0.1,font_size=8,node_size=50)
plt.show()
# plot degree distribution
plt.xscale('log')
plt.yscale('log')
plt.scatter(nxG.nodes,degree)
plt.show()

# ******
# Q 3 : pD => -0.29
# ******

pD = nx.degree_assortativity_coefficient(nxG)

# ******
# Q 4 : cc => 0.59
# ******
cc_dict = nx.clustering(nxG)
iter_cc = (cc_dict[i] for i in range(1, N))
cc = np.sum(np.fromiter(iter_cc, float)) / N

# ******
# Q 5 : EH => 1.93, Hmax => 5
# ******
iter_bn = (g.betweenness(i) for i in range(1, N))
betweenness = np.sum(np.fromiter(iter_bn, float))
EH = 2 * betweenness / (N * (N - 1))

Hmax = 0
for each in nx.shortest_path_length(nxG):
    for _, v in each[1].items():
        if v > Hmax:
            Hmax = v

# ******
# Q 6 : small_world(?)
# ******
# sigma = nx.sigma(nxG)
# small_world = sigma > 1

# ******
# Q 7 : max_adj_eigval => 60.63926551053449+0j
# ******
max_adj_eigval = sorted(nx.adjacency_spectrum(nxG))[-1]

# ******
# Q 8 : second_min_lap_eigval => 0.3810720817965757
# ******
second_min_lap_eigval = sorted(nx.laplacian_spectrum(nxG))[1]

# --------------------------------------------------------
'''