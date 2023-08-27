import numpy as np
from numpy import genfromtxt
import os
from sklearn.metrics import confusion_matrix
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
df = pandas.read_csv('FriendlyGameCorpus_3_4.csv',error_bad_lines=False)
my_data=df.values
del df

features=[]
internal=[]
internal.extend([my_data[0][2].replace(" ","")])
for i in range(1,int(my_data.shape[0])):
    if my_data[i][0]==my_data[i-1][0]:
        internal.extend([my_data[i][2].replace(" ","")])
    else:
        features.append(" ".join(internal))
        internal=[]
        internal.extend([my_data[i][2].replace(" ","")])
        
        
        
with open('game corpus.txt', 'w') as f:
    for item in features:
        f.write("%s\n" % item)
    
import gensim    
corpus = []
with open('game corpus.txt') as f:
    for i, line in enumerate(f):
        corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]))
        
        

      
train_corpus = list(corpus)    
     
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)      
model.build_vocab(train_corpus)      
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)      
  
     
ranks = []
second_ranks = []
for doc_id in range(200):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    sims = model.docvecs.most_similar([inferred_vector], topn=20)
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
    print(doc_id)

import collections    
collections.Counter(ranks)      

878181
f=[] 
for i in range(878181):
    if len(corpus[i][0])>30:
        f.append("".join(corpus[i][0]))
    print(i)

import csv     
with open('f.csv', 'w',newline='') as myfile:
      writer = csv.writer(myfile, delimiter=',')
      for line in f:
          writer.writerow([line])

df = pandas.read_csv('FriendlyGameCorpus_3_4_number.csv',error_bad_lines=False)
my_data=df.values
del df

import csv     
with open('fd.csv', 'w',newline='') as myfile:
      writer = csv.writer(myfile, delimiter=',')
      for line in df:
          writer.writerow([line])

import pandas as pd
df = pd.DataFrame(columns=['MasterID', 'Games'])
df['MasterID']=MasterID
df['Games']=features

df.to_csv('fd.csv')



features=[]
internal=[]
MasterID=[]
internal.extend([my_data[0][2]])
MasterID.extend([my_data[0][0]])
for i in range(1,int(my_data.shape[0])):
    if my_data[i][0]==my_data[i-1][0]:
        internal.extend([my_data[i][2]])
    else:
        features.append(internal)
        internal=[]
        internal.extend([my_data[i][2]])
        MasterID.extend([my_data[i][0]])
        
        
        
with open('game corpus.txt', 'w') as f:
    for item in features:
        f.write("%s\n" % item)
    
import gensim    
corpus = []
with open('game corpus.txt') as f:
    for i, line in enumerate(f):
        corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]))
      
train_corpus = list(corpus)    
     
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)      
model.build_vocab(train_corpus)      
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)      
  
     
ranks = []
second_ranks = []
for doc_id in range(200):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    #sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    sims = model.docvecs.most_similar([inferred_vector], topn=20)
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
    print(doc_id)