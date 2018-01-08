# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:51:36 2018

@author: pegasus
"""

from nltk.corpus import stopwords
import string
from collections import Counter
from os import listdir

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
    
def clean_doc(doc):
    tokens = doc.split()
    table = string.maketrans('', '')
    tokens= [w.translate(table, string.punctuation) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word)>1]
    return tokens
    
def create_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)
    
def process_docs(directory, vocab, is_trian):
    for filename in listdir(directory):
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        create_vocab(path, vocab)
    
    
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
vocab = Counter()
process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)

print(len(vocab))

min_occurance = 2
tokens = [k for k, c in vocab.items() if c>=min_occurance]
print(len(tokens))

save_list(tokens, 'vocab.txt')

