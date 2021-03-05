import torch
from torch.utils.data import DataLoader
import requests


class Config(dict):
    """
    Example:
    m = Config({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

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
