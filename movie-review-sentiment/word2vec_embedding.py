# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:11:42 2018

@author: pegasus
"""
from os import listdir
import string
from gensim.models import Word2Vec

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

def doc_to_clean_lines(doc, vocab):
    clean_lines = list()
    lines = doc.splitlines()
    for line in lines:
         tokens = line.split()
         table = string.maketrans('', '')
         tokens = [w.translate(table, string.punctuation) for w in tokens]
         tokens = [w for w in tokens if w in vocab]
         clean_lines.append(tokens)
    return clean_lines
    
def process_docs(directory, vocab, is_trian):
    lines = list()
    for filename in listdir(directory):
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
        lines += doc_lines
    return lines
    
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))

model = Word2Vec(sentences, size=100, window =5, workers = 8, min_count =1)
words = list(model.wv.vocab)
print('Vocab size : %d' % len(words))

filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary = False)