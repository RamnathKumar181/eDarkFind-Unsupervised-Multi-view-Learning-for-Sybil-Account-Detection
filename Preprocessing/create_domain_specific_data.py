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
    text.append((str(line)),username[i]])

col = ['text','vendor']
pd.DataFrame(text,columns=col).to_csv('All_sites_domain_specific.csv',index=None)
