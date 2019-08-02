import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
% matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

X = np.load('/content/gdrive/My Drive/eDarkFind/desc_eval_dataset.npy',allow_pickle=False)
sentences = []
for line in X:
  sentences.append(line[0])
  sentences.append(line[1])
print(len(set(sentences)))
uniq_sent = list(set(sentences))
# print(uniq_sent[0])
def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))
dic = {}
for index in range(len(uniq_sent)):
  print(index)
  temp = []
  for thing in uniq_sent[index].split(' <END> '):
    temp2 = []
    for ss in chunkstring(thing,512):
      marked_text = "[CLS] " + ss + " [SEP]"
    #   print(marked_text)
      tokenized_text = tokenizer.tokenize(marked_text)
      # print (tokenized_text)

      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

      # for tup in zip(tokenized_text, indexed_tokens):
      #   print (tup)
      segments_ids = [1] * len(tokenized_text)

      # Convert inputs to PyTorch tensors
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])
      with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    #   print ("Number of layers:", len(encoded_layers))
      layer_i = 0

    #   print ("Number of batches:", len(encoded_layers[layer_i]))
      batch_i = 0

    #   print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
      token_i = 0

    #   print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

      # Convert the hidden state embeddings into single token vectors

      # Holds the list of 12 layer embeddings for each token
      # Will have the shape: [# tokens, # layers, # features]
      token_embeddings = []

      # For each token in the sentence...
      for token_i in range(len(tokenized_text)):

        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):

          # Lookup the vector for `token_i` in `layer_i`
          vec = encoded_layers[layer_i][batch_i][token_i]

          hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)

      # Sanity check the dimensions:
    #   print ("Number of tokens in sequence:", len(token_embeddings))
    #   print ("Number of layers per token:", len(token_embeddings[0]))
      sentence_embedding = torch.mean(encoded_layers[11], 1)
      temp.append(np.asarray(sentence_embedding))
    temp2.append(np.mean(temp,axis=0)[0])
  dic[uniq_sent[index]] =np.mean(temp2,axis=0)
import pickle
output = open('../Embedding/Bert_Embedding.pkl', 'wb')
pickle.dump(dic, output)
output.close()
