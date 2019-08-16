# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:00:35 2019

@author: HP
"""
"""
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from json import writeJSON

n_steps_in, n_steps_out = 60, 30
n_features = 1
        

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

filepath = "wednesdayweights/text-gen-best.hdf5"       
model.load_weights(filepath)

import json
weights_list = model.get_weights()
#weightsjson={}
weightsarra=[]

datastore1 = json.dump(weights_list)
print(datastore1)
for i, weights in enumerate(weights_list):
    weightsjson={}
    weightsjson["layer"]=i
    dataaaaa=np.ndarray.tolist(weights)
    weightsjson["weightsdata"]=dataaaaa
    weightsarra.append(weightsjson)
   
    #writeJSON(weights)
for i,weights in enumerate(weights_list):


print(json.dumps(weights_list.tolist()))

import json
weights_list = model.get_weights()
print( json.dumps(weights_list.tolist()))

model.to_json()

mydict = {}
mydict["info"] = weights_list
anstring = str(mydict)
a=anstring.replace("'","\"");
returntoobj= json.loads(a)
returunobj2 = returntoobj["info"]

len(model.weights)

import json
"""

anlist = []
for i,weights in enumerate(weights_list):
    dataaa = np.ndarray.tolist(weights)
    anlist.append(dataaa)

print(json.dumps(anlist))

json_string = json.dumps(anlist)
datastore = json.loads(json_string)
anlist2 = []
for i,weights in enumerate(datastore):
    dataaa = np.array(weights)
    anlist2.append(dataaa)
 

