import sys

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import os
import pprint
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('words')
from nltk.corpus import words

def removeLineBreaks(tweet):
    return re.sub("\n\r|\r\n|\n|\r"," ", tweet)

def removeMultipleSpaces(tweet):
    return re.sub(" +", " ", tweet)

def lemmatizeTweet(tweet):
    return [WordNetLemmatizer().lemmatize(token) for token in word_tokenize(tweet)]

def removeAlphaNumeric(tweet):
    # return re.sub("[A-Za-z]+[0-9]+", "", tweet)
    return re.sub("[0-9A-Za-z]+", "", tweet)

def preprocess(tweet):
  tweet = removeLineBreaks(tweet)
  ll = lemmatizeTweet(tweet)
  temp = []
  for word in ll:
    if word in stop:
      temp.append(word)
    else:
      temp.append(removeAlphaNumeric(word))
  string = " ".join(temp)
  return removeMultipleSpaces(string)

pickle_in = open("../Data/All_sites.pickle","rb")
dict = pickle.load(pickle_in)
text = []
username = []
for key in dict:
    if type(key)==str:
        username.append(key)
for i in range(len(username)):
  print(i)
  for line in list(dict[username[i]]['Description'])[1:]:
    text.append([preprocess(str(line.lower())),username[i]])

col = ['text','vendor']
pd.DataFrame(text,columns=col).to_csv('All_sites_stylometric.csv',index=None)
