import pandas as pd
import csv
import numpy as np
import gc
import tensorflow as tf
import pickle
import tensorflow_hub as hub
from copy import deepcopy
from math import log


pickle_in = open("../Embeddings/Substance_Dictionary.pkl","rb")
dict1 = pickle.load(pickle_in)
keys = []
for key in dict1:
    keys.append(key)

X = np.load('../Data/subs_eval_dataset.npy',allow_pickle=False)
loc = ['Oxycodone', 'Heroin', 'Codeine', 'Tramadol', 'Buprenorphine', 'Hydrocodone', 'Hydromorphone', 'Kratom', 'Loperamide', 'Methadone', 'Morphine', 'Oxymorphone', 'Naloxone_type', 'Tapentadol', 'Fentanyl', 'Synthetic_Opioids']
total = {}
for item in loc:
    total[item]=0.0

pickle_in = open("../Data/All_sites.pickle","rb")
dict2 = pickle.load(pickle_in)

main = 0
for key in dict2:
    print(list(dict2[key]['Substance']))
    for item in list(dict2[key]['Substance']):
        if type(item)==str:
            print(item)
            for sp in item.split(' '):
                if sp in keys:
                    stuff= dict1[sp]
            print(stuff)
            total[stuff]+=1
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
        for sp in item.split(' '):
            if sp in keys:
                stuff= dict1[sp]
        print(stuff)
        dic1[stuff]=1.0
    temp1 = []
    for key in dic1:
        if(dic1[key]==0):
            temp1.append(0)
        else:
            temp1.append(-log(float(total[key])))
    final[line] = deepcopy(temp1)
    print(final[line])
print(len(final))
output = open('../Embeddings/Substance_Embeddings.pkl', 'wb')
pickle.dump(final, output)
output.close()
