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
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import TimeDistributed, Merge
from keras.layers.core import Lambda
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop
import keras.backend as K

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
    print(label+':')
    for w in x:
        print idx2word[w],
    print

i = 334
prt('H', Y_train[i])
prt('D', X_train[i])

i=334
prt('H', Y_test[i])
prt('D', X_test[i])

random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = maxlen, W_regularizer = regularizer, dropout = p_emb, weights =[embedding], mask_zero = True, name = 'embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size, recurrent_regularizer=regularizer, recurrent_dropout=0, bias_regularizer=regularizer, dropout=0, kernel_regularizer=regularizer, return_sequences = True, name = 'lstm_%d'%(i+1))
    model.add(lstm)
    model.add(Dropout(p_dense, name = 'dropout_%d'%(i+1)))
"""   
def simple_context(X, mask, n=activation_rnn_size, maxlend = maxlend, maxlenh = maxlenh):
    desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    activation_energies = K.reshape(activation_energies, (-1, maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))
"""
def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))    
class SimpleContext(Lambda):
    def __init__(self, **kwargs):
        super(SimpleContext, self).__init__(simple_context, **kwargs)
        self.supports_masking = True
        
    def compute_mask(self, input, input_mask = None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size-activation_rnn_size)
        return (nb_samples, maxlenh, n)
        
if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))
model.add(TimeDistributed(Dense(vocab_size, kernel_regularizer = regularizer, bias_regularizer= regularizer, name = 'timedistributed_1')))
model.add(Activation('softmax', name='activation_1'))
model.compile(loss='categorical_crossentropy', optimizer = optimizer)

