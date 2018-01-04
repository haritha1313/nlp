# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:10:12 2018

@author: pegasus
"""

from os import listdir
from pickle import dump

def load_doc(filename):
    file = open(filename)
    text = file.read()
    file.close()
    return text
    
def split_story(doc):
    index = doc.find('@highlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    high = ""
    highlights = [h.strip() for h in highlights if len(h) > 0]
    for high1 in highlights:
        if high1!=" ":
            high = high+" "+str(high1)
    #print(highlights)
    return story, high
    
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        doc = load_doc(filename)
        desc, head = split_story(doc)
        stories.append({'desc':desc, 'head':head, 'keywords':None})
    return stories
  
def clean_lines(lines):
    cleaned = ""
    #clean=list()
    for line in lines:
        index = line.find('CNN -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        line = line.split()
        #line = [word.lower() for word in line]
        line = [str(word) for word in line if word.isalpha()]
        line = [str(word) for word in line if len(word)>0]
        #line = [str(word) for word in line if len(word)>0]
        cleaned = cleaned + " " + str(' '.join(line))
    #cleaned = [c for c in cleaned if len(c) > 0]
    #clean.append(' '.join(cleaned))
    return cleaned
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded %d' % len(stories))

title = list()
description = list()
for sample in stories:
    #print(sample['head'])
    sample['desc'] = clean_lines(sample['desc'].split('\n'))
    sample['head'] = clean_lines(sample['head'].split('\n'))
    title.append(sample['head'])
    description.append(sample['desc'])
    
dump((title, description, None), open('tokens.pkl', 'wb'))
print('Saved %d titles and %d descriptions...' % (len(title), len(description)))