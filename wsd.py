import math
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
import re


def lcs(X, Y, m, n):
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]

    # Create a character array to store the lcs string
    lcs = [""] * (index + 1)
    lcs[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs

def wup_similarity(synset_i, synset_j, score=None, simulate_root=True):
    need_root = synset_i._needs_root()
    subsumers = synset_i.lowest_common_hypernyms(
        synset_j, simulate_root=simulate_root and need_root, use_min_depth=True
    )
    # If no LCS was found return None
    if len(subsumers) == 0:
        return 0
    subsumer = synset_i if synset_i in subsumers else subsumers[0]
    depth = subsumer.max_depth() + 1
    len1 = synset_i.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    len2 = synset_j.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    if len1 is None or len2 is None:
        return 0
    len1 += depth
    len2 += depth
    if score:
        log_score = math.log(score + 1)
        return ((2.0 * depth) + log_score) / ((len1 + len2) + log_score)
    else:
        return (2.0 * depth) / (len1 + len2)

def Gloss(synset):
    gloss = str(synset.definition())
    pattern = re.compile('[\W_]+')
    gloss = pattern.sub(' ', gloss)
    gloss = gloss.split(" ")
    return gloss

def Related(synset):
    l = []
    e = synset.hyponyms()
    if e:
        l.append(e)

    e = synset.hypernyms()
    if e:
        l.append(e)

    e = synset.member_meronyms()
    if e:
        l.append(e)

    e = synset.substance_meronyms()
    if e:
        l.append(e)

    e = synset.part_meronyms()
    if e:
        l.append(e)

    e = synset.part_holonyms()
    if e:
        l.append(e)

    e = synset.substance_holonyms()
    if e:
        l.append(e)

    e = synset.member_holonyms()
    if e:
        l.append(e)

    e = synset.also_sees()
    if e:
        l.append(e)

    e = synset.similar_tos()
    if e:
        l.append(e)

    l = [item for sublist in l for item in sublist]

    return l

def Lemma(synset):
    words = []
    for lemma in synset.lemmas():
        pattern = re.compile('[\W_]+')
        l = pattern.sub(' ', lemma.name())
        l = l.split(" ")
        for w in l:
            words.append(w)
    return words

def descriptor(synset):
    desc = []
    desc += Lemma(synset) + Gloss(synset)
    for r in Related(synset):
        desc += Gloss(r)
    desc = [x for x in desc if x not in stopwords.words('english')]
    desc.sort()
    return np.unique(desc)

def score(synset_i, synset_j):
    desc_i = descriptor(synset_i)
    desc_j = descriptor(synset_j)
    desc_i = list(filter(None, desc_i))
    desc_j = list(filter(None, desc_j))

    k = 0
    N = 0
    len_i_init = len(desc_i)
    len_j_init= len(desc_j)
    lcs_list = lcs(desc_i, desc_j, len_i_init, len_j_init)
    N += len(lcs_list)
    k += 1
    while len(lcs_list) > 1:
        for word in lcs_list:
            if word in desc_i and word in desc_j:
                desc_i.remove(word)
                desc_j.remove(word)
                len_i = len(desc_i)
                len_j = len(desc_j)
                lcs_list = lcs(desc_i, desc_j, len_i, len_j)
                N += len(lcs_list)+len(lcs_list)
                k += 1
    return N/k

def max_cj(ci, cjs):
    for cj in cjs:
        return wup_similarity(ci, cj, score(ci, cj))


def sentence_synsets(sentence, evaluate_neighbors=True, neighbors_d = 2):
    index = [i for i, x in enumerate(sentence) if x.lower() not in stopwords.words("english")]
    sentence = [x.lower() for x in sentence if x.lower() not in stopwords.words("english")]
    synsets = {}
    for i, (target, real_ind) in enumerate(zip(sentence, index)):
        if evaluate_neighbors:
            neighbors = []
            for x in range(1,neighbors_d):
                neg = (i - x) % len(sentence)
                pos = (i + x) % len(sentence)
                if neg < i:
                    neighbors.append(sentence[neg])
                else:
                    neighbors.append(sentence[(pos+1) % len(sentence)])
                if pos > i:
                    neighbors.append(sentence[pos])
                else:
                    neighbors.append(sentence[(neg-1)% len(sentence)])
        else:
            neighbors = sentence
        M = 0
        best_ci = None
        for ci in wn.synsets(target):
            max_sum = 0
            for other in neighbors:
                if other is not target:
                    for cj in wn.synsets(other):
                        wup = score(ci,cj)
                        max_sum += wup
            if max_sum > M:
                M=max_sum
                best_ci = ci
        if best_ci is not None:
            synsets[real_ind] = best_ci.name()
            print(target, best_ci.name())
    return synsets

