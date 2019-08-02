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

doc2vec = gensim.models.Doc2Vec.load('./User_domain_specific.model')
# data = np.asarray(pd.read_csv('./train_special_character_usage_user.csv', header=None))
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

output = open('../Embedding/Domain_Specific_Embedding.pkl', 'wb')
pickle.dump(dic, output)
output.close()
