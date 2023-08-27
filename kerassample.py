from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
import numpy as np
from numpy import genfromtxt
import os
from sklearn.metrics import confusion_matrix
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop')
df = pandas.read_csv('Billi_campaign.csv',error_bad_lines=False)
my_data=df.values
del df
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(my_data[:,0:3], dtype=float)
y=np.array(my_data[:,3], dtype=float)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# split into input (X) and output (y) variables
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6,activation='softmax'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, dummy_y, epochs=1000, batch_size=10)
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, dummy_y)
print('Accuracy: %.2f' % (accuracy*100))


