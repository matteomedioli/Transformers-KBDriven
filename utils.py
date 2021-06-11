import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
                    if isinstance(v, dict):
                        self[k] = Config(v)

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
                if isinstance(v, dict):
                    self[k] = Config(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]


def format_label(text):
    text = text.replace(".", "")
    for i, c in enumerate(text):
        try:
            int(c)
            root = text[0:i - 1]
            pos = text[i - 1]
            sense = text[i] + text[i + 1]
            lemma = ""
            if i + 2 != len(text):
                lemma = "." + text[i + 2:]
            return root + "." + pos + "." + sense + lemma
        except:
            next


def insert_between(iterable, fill, index):
    iterable = iter(iterable)
    for i, cur in enumerate(iterable):
        if i == index:
            yield fill
        yield cur


def cuda_setup():
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def ohe(nz_indices, device):
    i = torch.tensor([[0, 0, 0, 0], nz_indices.tolist()]).to(device)
    v = torch.tensor([1, 1, 1, 1]).to(device)
    return torch.sparse_coo_tensor(i, v, [1, 235233])


def get_batches(data, batch_size=32):
    return DataLoader(data, batch_size=batch_size)


def plot_pca(X, colors=None, n_components=3, element_to_plot=5000, path=None, epoch=0):
    if not colors:
        colors = np.ones(len(X))
    fig = plt.figure()
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca_components = pca.transform(X)
    if n_components == 2:
        plt.scatter(pca_components[:element_to_plot, 0],
                    pca_components[:element_to_plot, 1],
                    c=colors[:element_to_plot], s=1, marker='x')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_components[:element_to_plot, 0],
                   pca_components[:element_to_plot, 1],
                   pca_components[:element_to_plot, 2],
                   c=colors[:element_to_plot], s=1, marker='x')
    print("Saving png...")
    plt.savefig(path + str(epoch) + "e_pca.png")


def create_data_split(data, test_perc=0.1, val_perc=0.2):
    num_test = round(data.num_nodes * test_perc)
    num_val = round(data.num_nodes * val_perc)
    train_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    perm = torch.randperm(data.shape[0])
    t = int(data.shape[0]) - (num_val + num_test)
    train_mask[perm[:t]] = True
    val_mask[perm[t:t + num_test]] = True
    test_mask[perm[t + num_test:]] = True
    return train_mask, val_mask, test_mask


def load_model(path, checkpoint_fldr_and_bin, regularized=False, device='cuda'):
    state_dict = torch.load(path + checkpoint_fldr_and_bin, map_location=torch.device(device))
    keys = state_dict.keys()
    if regularized:
        for k in list(keys):
            if 'bert.' in k:
                state_dict[k[5:]] = state_dict[k]
                del state_dict[k]
    return BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=path,
        state_dict=state_dict)
