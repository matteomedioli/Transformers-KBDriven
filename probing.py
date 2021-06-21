# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals
from models import MLP
import sklearn
import os
import io
import copy
import logging
import numpy as np

assert (sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression


def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if 'noreg' not in config else config['noreg']
        self.config = config

    def run(self):
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        regs = [10 ** t for t in range(-5, -1)] if self.usepytorch else \
            [2 ** t for t in range(-2, 4, 1)]
        if self.noreg:
            regs = [1e-9 if self.usepytorch else 1e9]
        scores = []
        for reg in regs:
            if self.usepytorch:
                clf = MLP(self.classifier_config, inputdim=self.featdim,
                          nclasses=self.nclasses, l2reg=reg,
                          seed=self.seed, cudaEfficient=self.cudaEfficient)

                # TODO: Find a hack for reducing nb epoches in SNLI
                clf.fit(self.X['train'], self.y['train'],
                        validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(round(100 * clf.score(self.X['valid'],
                                                self.y['valid']), 2))
        logging.info([('reg:' + str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score \
            {1}'.format(optreg, devaccuracy))
        clf = LogisticRegression(C=optreg, random_state=self.seed)
        logging.info('Evaluating...')
        if self.usepytorch:
            clf = MLP(self.classifier_config, inputdim=self.featdim,
                      nclasses=self.nclasses, l2reg=optreg,
                      seed=self.seed, cudaEfficient=self.cudaEfficient)

            # TODO: Find a hack for reducing nb epoches in SNLI
            clf.fit(self.X['train'], self.y['train'],
                    validation_data=(self.X['valid'], self.y['valid']))
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X['train'], self.y['train'])

        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = round(100 * testaccuracy, 2)
        return devaccuracy, testaccuracy


class PROBINGEval(object):
    def __init__(self, task, task_path, seed=1111):
        self.seed = seed
        self.task = task
        logging.debug('***** (Probing) Transfer task : %s classification *****', self.task.upper())
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}
        self.loadFile(task_path)
        logging.info('Loaded %s train - %s dev - %s test for %s' %
                     (len(self.task_data['train']['y']), len(self.task_data['dev']['y']),
                      len(self.task_data['test']['y']), self.task))

    def do_prepare(self, params, prepare):
        samples = self.task_data['train']['X'] + self.task_data['dev']['X'] + \
                  self.task_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        self.tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                self.task_data[self.tok2split[line[0]]]['X'].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]['y'].append(line[1])

        labels = sorted(np.unique(self.task_data['train']['y']))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]['y']):
                self.task_data[split]['y'][i] = self.tok2label[y]

    def run(self, params, batcher):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.probing.batch_size
        logging.info('Computing embeddings for train/dev/test')
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.task_data[key]['X'], self.task_data[key]['y'] = map(list, zip(*sorted_data))

            task_embed[key]['X'] = []
            for ii in range(0, len(self.task_data[key]['y']), bsize):
                batch = self.task_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                task_embed[key]['X'].append(embeddings)
            task_embed[key]['X'] = np.vstack(task_embed[key]['X'])
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
        logging.info('Computed embeddings')

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        if self.task == "WordContent" and params.classifier['nhid'] > 0:
            config_classifier = copy.deepcopy(config_classifier)
            config_classifier['classifier']['nhid'] = 0
            print(params.classifier['nhid'])

        clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['dev']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['dev']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, self.task.upper()))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(task_embed['dev']['X']),
                'ntest': len(task_embed['test']['X'])}


"""
Surface Information
"""


class LengthEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'sentence_length.txt')
        # labels: bins
        PROBINGEval.__init__(self, 'Length', task_path, seed)


class WordContentEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'word_content.txt')
        # labels: 200 target words
        PROBINGEval.__init__(self, 'WordContent', task_path, seed)


"""
Latent Structural Information
"""


class DepthEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'tree_depth.txt')
        # labels: bins
        PROBINGEval.__init__(self, 'Depth', task_path, seed)


class TopConstituentsEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'top_constituents.txt')
        # labels: 'PP_NP_VP_.' .. (20 classes)
        PROBINGEval.__init__(self, 'TopConstituents', task_path, seed)


class BigramShiftEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'bigram_shift.txt')
        # labels: 0 or 1
        PROBINGEval.__init__(self, 'BigramShift', task_path, seed)


# TODO: Voice?

"""
Latent Semantic Information
"""


class TenseEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'past_present.txt')
        # labels: 'PRES', 'PAST'
        PROBINGEval.__init__(self, 'Tense', task_path, seed)


class SubjNumberEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'subj_number.txt')
        # labels: 'NN', 'NNS'
        PROBINGEval.__init__(self, 'SubjNumber', task_path, seed)


class ObjNumberEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'obj_number.txt')
        # labels: 'NN', 'NNS'
        PROBINGEval.__init__(self, 'ObjNumber', task_path, seed)


class OddManOutEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'odd_man_out.txt')
        # labels: 'O', 'C'
        PROBINGEval.__init__(self, 'OddManOut', task_path, seed)


class CoordinationInversionEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'coordination_inversion.txt')
        # labels: 'O', 'I'
        PROBINGEval.__init__(self, 'CoordinationInversion', task_path, seed)
