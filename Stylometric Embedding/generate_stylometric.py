from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import csv
import gensim, os
import pickle
import pandas as pd
import numpy as np
import os
import difflib
import pprint
import pickle
import textdistance
from sklearn import preprocessing
import math
import scipy
import pandas as pd
import base64
import numpy as np
# import imageio
import os
import scipy
import gensim
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import gensim.corpora as corpora
import itertools
import nltk
import pickle
from copy import deepcopy
import sys

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
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('words')
from nltk.corpus import words
print("32fs" in words.words())

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

doc2vec = gensim.models.Doc2Vec.load('./User_stylometric.model')
DIM = 300

X = np.load('../Data/desc_eval_dataset.npy',allow_pickle=False)
sentences = []
for line in X:
  sentences.append(line[0])
  sentences.append(line[1])
# print(len(set(sentences)))

uniq_sent = list(set(sentences))
# print(uniq_sent[0])

dic = {}
for index in range(len(uniq_sent)):
	# print(index)
    string = uniq_sent[index]
    # print(string)
    # if(type(string)==np.str):
    final = string.split("<END>")

    vectors = np.asarray(doc2vec.infer_vector(final))
    dic[uniq_sent[index]] = vectors

output = open('../Embedding/Stylometric_Sentence.pkl', 'wb')
pickle.dump(dic, output)
output.close()
