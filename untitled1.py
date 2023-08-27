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
        
corpus = []
with open('game corpus.txt') as f:
    line = f.readline()
    while line:
       words = [x for x in line.split()]
       corpus.append(words)
       line = f.readline()
           
num_of_sentences = len(corpus)
num_of_words = 0
for line in corpus:
    num_of_words += len(line)

print('Num of sentences - %s'%(num_of_sentences))
print('Num of words - %s'%(num_of_words))
size = 50
window_size = 5 # sentences weren't too long, so
epochs = 10
min_count = 2
workers = 6

from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from pandas import Timedelta
from datetime import timedelta
# train word2vec model using gensim
#modelnonptm = Word2Vec(corpus, sg=1,window=window_size,size=size,
#                 min_count=min_count,workers=workers,iter=epochs,sample=0.01)

from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):

     def __init__(self):
         self.epoch = 0

     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch))
               
     def on_epoch_end(self, model):
         print("Epoch #{} end".format(self.epoch))
         self.epoch += 1

epoch_logger = EpochLogger()
#w2v_model = Word2Vec(corpus, iter=5, size=10, min_count=0, seed=42)
w2v_model=Word2Vec(corpus, sg=1,window=window_size,size=size,min_count=min_count,workers=workers,iter=epochs,callbacks=[epoch_logger])










model.save('w2v_model')