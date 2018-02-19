# Karan Gurazada, Siddharth Srinivasan
# Please note that Tensorflow best works on cuda enabled gpu
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys


f = open('QS', 'r',encoding = 'latin-1')
c = ''
s = ''
data = {}
for i in range(28910):
    x = f.readline()
    if ('--' in x):
        x = x.split('--')
        c = x[1]
    else:
        s = x
    if (c != '' and s != ''):
        try:
            blah = data[c]
        except:
            data[c] = []
        data[c].append(s)
        c = ''
        s = ''



tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)
stemmer = LancasterStemmer()
#data = None
#with open('dump.json') as json_data:
#    data = json.load(json_data)
    #print(data)
categories = list(data.keys())
words = []
docs = []
for each_category in data.keys():
    for each_sentence in data[each_category]:
        each_sentence = remove_punctuation(each_sentence)
        #print(each_sentence)
        w = nltk.word_tokenize(each_sentence)
        #print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
#print(words)
#print(docs)
print(categories)
training = []
output = []
output_empty = [0] * len(categories)
for doc in docs:
    bow = []
    token_words = doc[0]
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1
    training.append([bow, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=100, batch_size=25, show_metric=True)
model.save('model.tflearn')
#model.load('model.tflearn')
def get_tf_record(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))
for i in range(1,len(sys.argv)):
    print(categories[np.argmax(model.predict([get_tf_record(sys.argv[i])]))])
