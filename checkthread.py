# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:52:45 2019

@author: HP
"""
import pandas as pd
from numpy import array
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import date,datetime,timedelta
import psycopg2
import json
import ast

tempday=-1
t1 = datetime.now()
weekd = t1.weekday()
anlist = []
anlist2 = []
while(True):
    if(tempday != weekd):
        print('in the loop')
        tempday = weekd
        
    t1 = datetime.now()
    if t1.hour == 00 and t1.minute == 00 and t1.second == 00:
        weekd = t1.weekday() 
        time.sleep(5)
        anlist.append('Hello')
    elif t1.min == 2 and t1.second == 00:
        anlist2.append('hii')
        print('Hello')
        time.sleep(5)