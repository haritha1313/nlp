# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:03:00 2018

@author: pegasus
"""
import os
import keras
import pickle
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

FN = 'train'
FN0 = 'vocabulary-embedding'
FN1 = 'train'

maxlend = 50
maxlenh = 50
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3
batch_norm = False

activation_rnn_size = 80 if maxlend else 0

seed = 42
p_W, p_U, p_dense, p_emb, weight_decay = 0,0,0,0,0
optimizer = 'adam'
LR = 1e-4
batch_size = 64
nflips = 10

nb_train_samples = 30000
nb_val_samples = 3000

with open('%s.pkl'%FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    
vocab_size, embedding_size = embedding.shape

with open('%s.data.pkl'%FN0, 'rb') as fp:
    X, Y = pickle.load(fp)
    
nb_unknown_words = 10

print('number of examples', len(X), len(Y))
print('dimension of embedding space for words', embedding_size)
print('vocabulary size', vocab_size, 'the last %d words can be used as placeholders for unknown or oov words' % nb_unknown_words)
print('tital number of different words', len(idx2word), len(word2idx))
print('number of words outside vocabulary which we can substitute using glove similarityy', len(glove_idx2idx))
print('number of wordss that will be regarded as unknown(unk)/out-of-vocabulary(oov)', len(idx2word)-vocab_size-len(glove_idx2idx))


for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i
    
oov0 = vocab_size-nb_unknown_words
for i in range(oov0, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = nb_val_samples, random_state = seed)
print(len(X_train), len(Y_train), len(X_test), len(Y_test))

empty= 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

def prt(label, x):
    print(label+':', )
    for w in x:
        print(idx2word[w], )
    print( )

i = 334
prt('H', Y_train[i])
prt('D', X_train[i])