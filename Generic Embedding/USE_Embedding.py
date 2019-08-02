import pandas as pd
import csv
import numpy as np
import gc
import tensorflow as tf
import pickle
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
# messages = ["That band rocks!That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool.That song is really cool."]
X = np.load('desc_eval_dataset.npy',allow_pickle=False)
sentences = []
for line in X:
  sentences.append(line[0])
  sentences.append(line[1])
print(len(set(sentences)))
uniq_sent = list(set(sentences))
# print(uniq_sent[0])

dic = {}
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  for index in range(len(uniq_sent)):
    print(index)
    temp = []
    for thing in uniq_sent[index].split(' <END> '):
        temp.append(session.run(embed([thing])))
    dic[uniq_sent[index]] =np.mean(temp,axis=0)

output = open('USE_Sentence_dic.pkl', 'wb')
pickle.dump(dic, output)
output.close()
