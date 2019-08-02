import pandas as pd
import csv
import numpy as np
import gc
import tensorflow as tf
import pickle
import tensorflow_hub as hub
from copy import deepcopy
from math import log


pickle_in = open("../Embeddings/Location_Dictionary.pkl","rb")
dict1 = pickle.load(pickle_in)
keys = []
for key in dict1:
    keys.append(key)

X = np.load('../Data/shipsfr_eval_dataset.npy',allow_pickle=False)
loc = ['US', 'Worldwide', 'AU', 'UK', 'SE', 'Other', 'EU', 'CA', 'NA', 'HR', 'MG', 'DE', 'OC', 'SW', 'IE', 'NE', 'FI', 'YE', 'CH', 'NZ', 'NO', 'IN', 'NY', 'PL', 'FR', 'HK', 'AS', 'EE', 'DK', 'AF', 'SA', 'WF', 'LA', 'CN', 'FN', 'BE', 'AT', 'NL', 'PH', 'GR', 'PE', 'SG', 'PK', 'KH', 'JP', 'IT', 'UA', 'NP', 'GB', 'MX', 'PT', 'ES', 'TH', 'ML', 'CO', 'SK', 'ID', ' AU', 'RU', 'LV', 'USA', 'Au', 'AZ', 'CZ', 'TJ', 'GE', 'SI']
total = {}
for item in loc:
    total[item]=0.0

pickle_in = open("../Data/All_sites.pickle","rb")
dict2 = pickle.load(pickle_in)

main = 0
for key in dict2:
    print(list(dict2[key]['Ships from']))
    for item in list(dict2[key]['Ships from']):
        if type(item)==str:
            for items in dict1[item]:
                total[items]+=1
                main+=1
for key in total:
    total[key]/=main
print(total)

sentences = []

for line in X:
  if line[0] not in sentences:
      sentences.append(line[0])
  if line[1] not in sentences:
      sentences.append(line[1])
print(len(sentences))

final = {}
for line in sentences:
    dic1 = {}
    for item in loc:
        dic1[item]=0.0
    for item in line.split(' <END> '):
        for items in dict1[item]:
            dic1[items]=1.0
    temp1 = []
    for key in dic1:
        if(dic1[key]==0):
            temp1.append(0)
        else:
            temp1.append(-log(float(total[key])))
    final[line] = deepcopy(temp1)
    print(final[line])
print(len(final))
output = open('../Embeddings/Location_Embeddings.pkl', 'wb')
pickle.dump(final, output)
output.close()
