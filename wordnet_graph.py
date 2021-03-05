from nltk.corpus import wordnet
from utils import format_label
import networkx as nx
import timeit
import time
from IPython.display import clear_output
import torch
import torch_geometric
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


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
        wordnet_data.root = T[0:i] + T[i + 1:]
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


def create_text_gnn(layer_info, dataset):
    lyr_dims = parse_as_int_list(layer_info["layer_dim_list"])
    lyr_dims = [dataset.node_feats.shape[1]] + lyr_dims
    weights = None
    if layer_info["class_weights"].lower() == "true":
        counts = Counter(dataset.label_inds[dataset.node_ids])
        weights = len(counts) * [0]
        min_weight = min(counts.values())
        for k, v in counts.items():
            weights[k] = min_weight / float(v)
        weights = torch.tensor(weights, device=FLAGS.device)

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
