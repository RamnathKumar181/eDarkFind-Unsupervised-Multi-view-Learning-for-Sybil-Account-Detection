import pandas as pd
import csv
import numpy as np
import gc
import tensorflow as tf
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

X = np.load('Data/loc_eval_dataset.npy',allow_pickle=False)

pickle_in = open("Embeddings/Loc_Embeddings.pkl","rb")
dict1 = pickle.load(pickle_in)
n = []
for key in dict1:
    n.append(key)
print(len(n))
#
pred = []

for line in X:
    pred.append([abs(cosine_similarity(np.asarray(dict1[line[0]]).reshape(1,-1),np.asarray(dict1[line[1]]).reshape(1,-1))[0][0]),int(line[2])])
pd.DataFrame(pred).to_csv('Loc.csv',index = None)

data = pd.read_csv('Loc.csv')
T=data.values
X = T[:,0]
y = T[:,1]
print(abs(X))
y_pred =[]
for line in X:
    if(abs(line)>0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print()
print("Classification Report")
print(classification_report(y, y_pred))
