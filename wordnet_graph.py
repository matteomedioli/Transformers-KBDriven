import logging
import timeit
from itertools import chain

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from IPython.display import clear_output
from nltk.corpus import wordnet
from torch_geometric.data import InMemoryDataset, Data, download_url
from tqdm import tqdm

from utils import format_label

relations_to_ids = {
    'synset': 0,
    'lemma': 1,
    'hypernym': 2,
    'hyponym': 3,
    'antonym': 4,
    'pertainym': 5,
    'member_holonym': 6,
    'member_meronym': 7,
    'part_holonym': 8,
    'part_meronym': 9,
    'cause': 10,
    'also_see': 11,
    'derivationally_related_form': 12,
    'entailment': 13,
    'region_domain': 14,
    'similar_tos': 15,
    'substance_holonym': 16,
    'substance_meronym': 17,
    'topic_domain': 18,
    'usage_domain': 19,
    'verb_group': 20
}


class WordNetGraph:
    '''
    Generate a NetworkX graph to convert in Data Pytorch Geometric
    dataset using torch_geometric.utils.convert.from_networkx

    Attributes:
        all_synset          Wordnet Synsets list from nltk
        relation_names      Edge labels list for relation between Synset and Lemma nodes
        relation_functions  Functions list to retrieve related nodes given a single node
        graph               NetwrokX graph

        NB: embeddings was subsequently generated in form
                        [root.pos.sense.lemma]
            sizes of sets [86571,5,59,148598] = 235233 (OHE)

    '''

    def __init__(self, graph="data/graphs/wordnet.gpickle"):

        self.n_id = -1
        # List to store ordered node ids
        # i.e Index 0: dog.n.01, Index 1: dog.n.01.dog ...
        self.node_ids = []

        self.all_synset = list(wordnet.all_synsets())

        self.relation_names = ["hypernym",
                               "hyponym",
                               "part_meronym",
                               "substance_meronym",
                               "member_meronym",
                               "part_holonym",
                               "substance_holonym",
                               "member_holonym",
                               "topic_domain",
                               "region_domain",
                               "usage_domain",
                               "entailment",
                               "cause",
                               "also_see",
                               "verb_group",
                               "similar_tos",
                               "antonym",
                               "derivationally_related_form",
                               "pertainym"]

        self.relation_functions = [lambda s: s.hypernyms(),
                                   lambda s: s.hyponyms(),
                                   lambda s: s.part_meronyms(),
                                   lambda s: s.substance_meronyms(),
                                   lambda s: s.member_meronyms(),
                                   lambda s: s.part_holonyms(),
                                   lambda s: s.substance_holonyms(),
                                   lambda s: s.member_holonyms(),
                                   lambda s: s.topic_domains(),
                                   lambda s: s.region_domains(),
                                   lambda s: s.usage_domains(),
                                   lambda s: s.entailments(),
                                   lambda s: s.causes(),
                                   lambda s: s.also_sees(),
                                   lambda s: s.verb_groups(),
                                   lambda s: s.similar_tos(),
                                   lambda s: s.antonyms(),
                                   lambda s: s.derivationally_related_forms(),
                                   lambda s: s.pertainyms()]

        if graph:
            self.graph = nx.read_gpickle(graph)
            self.node_ids_init()
        else:
            self.graph = nx.DiGraph()
            self.init_graph()
        self.node_ids = [format_label(x) if x.count(".") > 3 else x for x in self.node_ids]
        logging.info("[WORDNET LOAD COMPLETE]")

    def node_id(self):
        self.n_id += 1
        return self.n_id

    # Wrapper class for Synsets
    class SynsetNode:
        def __init__(self, outer, synset):
            self.node_id = outer.node_id()
            self.name = synset.name()
            self.offset = synset.offset()
            self.pos = synset.pos()
            self.root = self.name.split(".")[0]
            self.sense = self.name.split(".")[2]
            self.lexname = synset.lexname()
            self.definition = synset.definition()
            self.embedding = None

    # Wrapper class for Lemmas
    class LemmaNode:
        def __init__(self, outer, synset, lemma):
            self.node_id = outer.node_id()
            self.name = synset.name() + "." + lemma.name()
            self.root = self.name.split(".")[0]
            self.pos = self.name.split(".")[1]
            self.sense = self.name.split(".")[2]
            self.key = lemma.key()
            self.lang = lemma.lang()
            self.lemma = self.name.split(".")[3]
            self.embedding = None

    def SynsetNote(self, synset):
        return WordNetGraph.SynsetNode(self, synset)

    def LemmaNode(self, synset, lemma):
        return WordNetGraph.LemmaNode(self, synset, lemma)

    def init_graph(self):
        '''
        Initialize graph:
        1. Generate all nodes using wrappers
        2. Generates all relations between added nodes using saved index in node_ids
        '''

        # Synsets and lemmas
        for synset in self.all_synset:
            # Create a SynsetNode containing synset information
            syn_node = self.SynsetNode(synset)
            # Add synset node to graph
            self.graph.add_node(syn_node.node_id,
                                node_type="synset",
                                root=syn_node.root,
                                pos=syn_node.pos,
                                sense=syn_node.sense,
                                name=syn_node.name,
                                offset=syn_node.offset,
                                lexname=syn_node.lexname,
                                definition=syn_node.definition,
                                embedding=syn_node.embedding
                                )
            # Add synset node to node_ids list
            self.node_ids.append(synset.name())
            for lemma in synset.lemmas():
                lemma_node = self.LemmaNode(synset, lemma)
                self.graph.add_node(lemma_node.node_id,
                                    node_type="lemma",
                                    root=lemma_node.root,
                                    pos=lemma_node.pos,
                                    sense=lemma_node.sense,
                                    lemma=lemma_node.lemma,
                                    name=lemma_node.name,
                                    key=lemma_node.key,
                                    lang=lemma_node.lang,
                                    embedding=lemma_node.embedding)
                self.node_ids.append(synset.name() + "." + lemma.name())
                self.graph.add_edge(syn_node.node_id, lemma_node.node_id, relation="lemma")
                self.graph.add_edge(lemma_node.node_id, syn_node.node_id, relation="synset")
        print(len(self.all_synset))
        for i, synset in enumerate(self.all_synset):
            start = timeit.default_timer()
            print(i, "/", len(self.all_synset))
            for rel_name, fn in zip(self.relation_names, self.relation_functions):
                try:
                    for related_synset in fn(synset):
                        self.graph.add_edge(list(self.graph.nodes())[self.synset_node_index(synset)],
                                            list(self.graph.nodes())[self.synset_node_index(related_synset)],
                                            relation=rel_name)
                except:
                    try:
                        for lemma in synset.lemmas():
                            for related_lemma in fn(lemma):
                                self.graph.add_edge(list(self.graph.nodes())[self.lemma_node_index(lemma)],
                                                    list(self.graph.nodes())[self.lemma_node_index(related_lemma)],
                                                    relation=rel_name)
                    except:
                        next
            stop = timeit.default_timer()
            print('Time for node: ', stop - start)
            clear_output()
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        nx.write_gpickle(self.graph, "final.gpickle")
        print("SAVED!")

    def synset_node_index(self, synset):
        return self.node_ids.index(synset.name())

    def lemma_node_index(self, lemma):
        return self.node_ids.index(lemma.synset().name() + "." + lemma.name())

    def node_ids_init(self):
        for synset in self.all_synset:
            self.node_ids.append(synset.name())
            for lemma in synset.lemmas():
                self.node_ids.append(synset.name() + "." + lemma.name())

    def get_node_id(self, target):
        # TODO: Definire un euristica di decisione migliore per il nodo target
        return self.lemma_node_index(wordnet.lemmas(target, wordnet.NOUN)[0])

    def subgraph(self, node_id, max_depth=None):
        if not max_depth:
            max_depth = len(self.graph)
        nodes = list(nx.dfs_postorder_nodes(self.graph, node_id, max_depth))
        node_ids = [
            self.node_ids[i].split(".") if len(self.node_ids[i].split(".")) == 4 else self.node_ids[i].split(".") + [
                "None"] for i in nodes]
        df = pd.DataFrame(node_ids, columns=["root", "pos", "sense", "lemma"])
        return len(np.unique(df["root"])), len(np.unique(df["lemma"])), len(np.unique(df["root"])) + len(
            np.unique(df["lemma"]))


def reduce_wordnet(wordnet_data):
    lemmas = [i for i, t in enumerate(wordnet_data.node_type) if t == 1]
    c = 0
    for o, i in enumerate(lemmas):
        i = i - c
        T = wordnet_data.edge_index
        wordnet_data.edge_index = torch.stack(
            [torch.cat([T[0][0:i], T[0][1 + i:]]), torch.cat([T[1][0:i], T[1][i + 1:]])])
        T = wordnet_data.name
        wordnet_data.name = T[0:i] + T[i + 1:]
        T = torch.Tensor(wordnet_data.node_type)
        wordnet_data.node_type = torch.cat([T[0:i], T[i + 1:]])
        T = wordnet_data.pos
        wordnet_data.pos = torch.cat([T[0:i], T[i + 1:]])
        T = wordnet_data.relation
        wordnet_data.relation = T[0:i] + T[i + 1:]
        T = wordnet_data.root
        wordnet_data.root = torch.cat([T[0:i], T[i + 1:]])
        T = wordnet_data.sense
        wordnet_data.sense = torch.cat([T[0:i], T[i + 1:]])
        c += 1
    return wordnet_data


def plot_deepth_nodes_number(graph):
    data = []
    x = [500, 800, 1000, 2000, 3000]
    for i in x:
        data.append(graph.subgraph(0, i))
    plt.plot(x, data)
    plt.xticks(x)
    plt.legend(["unique root", "unique lemma", "total"], loc=7)
    plt.xlabel("DFS Deepth - wordnode target")
    plt.ylabel("Number of Nodes (ohe dim)")
    plt.title("Number of unique values to embed")
    plt.show()


def networkx_to_Data(G):
    r"""Converts a :obj:`networkx.Graph` G or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    synset_feats = dict(G.nodes[0]).keys()
    lemma_feats = dict(G.nodes[1]).keys()
    features = synset_feats | lemma_feats
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}
    for f in features:
        data[f] = []

    print(len(G.nodes()))
    for i, (_, feat_dict) in tqdm(enumerate(G.nodes(data=True))):
        for key, value in feat_dict.items():
            if key not in feat_dict.keys():
                data[str(key)] = ["-"] if i == 0 else data[str(key)] + ["-"]
            else:
                data[str(key)] = [value] if i == 0 else data[str(key)] + [value]
    print(len(G.edges()))
    for i, (_, _, feat_dict) in tqdm(enumerate(G.edges(data=True))):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


class WordNet18RR(InMemoryDataset):
    r"""The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
    Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
    entities, 11 relations and 93,003 fact triplets.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original')

    edge2id = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }
    node2id = {}
    idx = 0

    def __init__(self, root, transform=None, pre_transform=None):
        super(WordNet18RR, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self):
        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path, 'r') as f:
                data = f.read().split()

                src = data[::3]
                dst = data[2::3]
                edge_type = data[1::3]

                for i in chain(src, dst):
                    if i not in self.node2id:
                        self.node2id[i] = self.idx
                        self.idx += 1

                src = [self.node2id[i] for i in src]
                dst = [self.node2id[i] for i in dst]
                edge_type = [self.edge2id[i] for i in edge_type]

                srcs.append(torch.tensor(src, dtype=torch.long))
                dsts.append(torch.tensor(dst, dtype=torch.long))
                edge_types.append(torch.tensor(edge_type, dtype=torch.long))

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[:srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0):srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0):] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_filter(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# WN = WordNet18RR("/home/med/Scrivania/data/wordnet/")
# WN.process()
# print(WN.node2id["dog.n.01"])
#
# names = [x.split(".")[0] for x in WN.node2id.keys()]
# pos = [x.split(".")[1] for x in WN.node2id.keys()]
# sense = [x.split(".")[2] for x in WN.node2id.keys()]
#
# print(len(np.unique(names)))
# print(np.unique(pos))
# print(np.unique(sense))

data = torch.load("synsets.pt")
print(data)
