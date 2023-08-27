# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:55:22 2020

@author: mhosseini
"""

import sys, os
sys.path.append(os.pardir)


import tensorflow as tf
import json
import argparse


from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout
from keras.callbacks import TensorBoard



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='char_cnn_zhang', help='Specifies which model to use: char_cnn_zhang or char_cnn_kim')
FLAGS = parser.parse_args(["--model", "char_cnn_zhang"])

# Load configurations
config = json.load(open('../config.json'))

# change key from 'model' to 'char_cnn_zhang'
model_name = config['model'] # char_cnn_zhang
config['model'] = config[model_name]

# Set the data path in order to run in the notebook 
config['data']["training_data_source"] = '../data/ag_news_csv/train.csv'
config['data']["validation_data_source"] = '../data/ag_news_csv/test.csv'

# Load training data
training_data = Data(data_source=config["data"]["training_data_source"],
                     alphabet=config["data"]["alphabet"],
                     input_size=config["data"]["input_size"],
                     num_of_classes=config["data"]["num_of_classes"])
training_data.load_data()
training_inputs, training_labels = training_data.get_all_data()

# Load validation data
validation_data = Data(data_source=config["data"]["validation_data_source"],
                       alphabet=config["data"]["alphabet"],
                       input_size=config["data"]["input_size"],
                       num_of_classes=config["data"]["num_of_classes"])
validation_data.load_data()
validation_inputs, validation_labels = validation_data.get_all_data()