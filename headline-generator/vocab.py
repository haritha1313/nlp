# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:05:14 2018

@author: pegasus
"""

import pickle
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np

func = 'vocabulary-embedding'
seed = 42
vocab_size = 40000
embedding_dim = 100
lower = False
func1 = 'tokens'
with open('%s.pkl' % func1, 'rb') as fp:
    heads, desc, keywords = pickle.load(fp)
    
if lower:
    heads = [h.lower() for h in heads]
    desc = [d.lower() for d in desc]
    
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x:x[0], sorted(vocabcount.items(), key = lambda x: -x[1]))
    return vocab, vocabcount
    
vocab, vocabcount = get_vocab(heads+desc)

"""
plt.plot([vocabcount[w] for w in vocab]);
plt.gca().set_xscale("log", nonposx = 'clip')
plt.gca().set_yscale("log", nonposy = 'clip')
plt.title('word distribution in data')
plt.xlabel('rank')
plt.ylabel('total appearances');
plt.show()
"""

empty = 0
eos = 1
start_idx = eos+1

def get_idx(vocab):
    word2idx = dict((word, idx+start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>']=empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx, word) for word, idx in word2idx.iteritems())
    return word2idx, idx2word
    
word2idx, idx2word = get_idx(vocab)

fname = 'glove.6B.%dd.txt' % embedding_dim

datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
datadir = os.path.join(datadir_base, 'datasets')
glove_name = os.path.join(datadir, fname)

process = subprocess.Popen(['wc', '-l', glove_name], stdout=subprocess.PIPE)
glove_n_symbols = int(process.communicate()[0].split()[0])

glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
global_scale = .1
with open(glove_name, 'r') as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = map(float, l[1:])
        i += 1
glove_embedding_weights *= global_scale

for w, i in glove_index_dict.iteritems():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i
        

np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2
embedding = np.random.uniform(low = -scale, high = scale, size = shape)
print('random-embedding/glove scale', scale, 'std', embedding.std())

c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'):
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1
print('number of tokens, in small vocab, found in glove copied to embedding', c, c/float(vocab_size))

glove_thr = 0.5
word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g
    
normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w, idx in word2idx.iteritems():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx, :].copy()
        gweight /= np.sqrt(np.dot(gweight, gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s<glove_thr:
                break
            if idx2word[embedding_idx] in word2glove:
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x : -x[2])
#print('# of substitutes found', len(glove_match))

glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)

Y = [[word2idx[token] for token in headline.split()] for headline in heads]
print(len(Y))

plt.hist(map(len, Y), bins =50);
#plt.show()

X = [[word2idx[token] for token in d.split()] for d in desc]
print(len(X))

plt.hist(map(len,X),bins=50);
#plt.show()

with open('%s.pkl'%func, 'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, -1)
    
with open('%s.data.pkl'%func, 'wb') as fp:
    pickle.dump((X,Y), fp, -1)
