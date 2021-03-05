import html
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import logging
import linecache


def clear_text(original, output):
    file1 = open(str(original, "r"))
    file2 = open(str(output + ".txt"), "w")
    for line in tqdm(file1.readlines()):
        line = html.unescape(line)
        file2.write(line)
    file1.close()
    file2.close()


class WikiCorpora:
    def __init__(self, batch_size, dump_file):
        self.bookmark = 0
        self.batch_size = batch_size
        self.dump_file = dump_file

    def generate_batch(self):
        # TODO: return the last batch. Now it's not included cause if len(corpora) == self.batch_size: yield
        corpora = []
        with open(self.dump_file) as f:
            for i, line in enumerate(f):
                if len(corpora) < self.batch_size:
                    if i >= self.bookmark:
                        corpora.append(line)
                        if len(corpora) == self.batch_size:
                            res = corpora
                            corpora = []
                            self.bookmark = self.bookmark + self.batch_size
                            yield res

    def reset(self, batch_size=0):
        self.bookmark = 0
        self.batch_size = batch_size
