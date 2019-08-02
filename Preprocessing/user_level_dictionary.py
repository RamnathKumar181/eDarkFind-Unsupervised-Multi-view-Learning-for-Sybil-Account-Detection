import pandas as pd
import numpy as np
import os
import difflib
import pprint
import pickle

data = pd.read_csv('Data/All_sites.csv')
X = data.values
p=[]
for line in X:
    p.append(line[0])
print(len(set(p)))
dic = {}
for key in set(p):
    temp= {}
    sub=[]
    shipto=[]
    shipfrom=[]
    desc=[]
    for line in X:
        if(line[0]==key):
            sub.append(line[1])
            shipto.append(line[3])
            shipfrom.append(line[2])
            desc.append(line[4])
    temp['Substance'] = set(sub)
    temp['Ships to'] = set(shipto)
    temp['Ships from'] = set(shipfrom)
    temp['Description'] = set(desc)
    if len(str(key))!=0:
        if(str(key)[0]==' ' or str(key)[0]=='@'):
            if len(str(key))>2:
                dic[key[1:]] = temp
        else:
            dic[key] = temp
pickle_out = open("Data/All_sites.pickle","wb")
pickle.dump(dic, pickle_out)
pickle_out.close()
