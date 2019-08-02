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

A = np.load('wgcca_embeddings2.npz')
y = A["G.npy"]
w = A["ids.npy"]
data = pd.read_csv('Key_wgca_loc.tsv', sep="\t")
X = data.values
# uniq_sent = list(set(sentences))
dic = {}
for index in w:
    dic[X[int(index)-1][1]] = y[int(index)-1]
# for index in range(len(uniq_sent)):
#     for index2 in range(len(w)):
#         if w[index2]==uniq_sent[index]:
#             dic[w[index2]] = y[index]
#             break
# for index in range(len(uniq_sent)):
#     print(dic[uniq_sent[index]])
output = open('Embeddings/Loc_Embeddings.pkl', 'wb')
pickle.dump(dic, output)
output.close()
