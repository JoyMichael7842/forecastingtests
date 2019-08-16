# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:20:28 2019

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


def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


n_steps_in, n_steps_out = 30, 15
n_features = 1
        

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

filepath = "thurweights/text-gen-best.hdf5"

model.load_weights(filepath)
plist = []
 
while(True):
    t1 = datetime.now()
    if t1.weekday() == 3:
        
        if(t1.hour == 00 and t1.minute == 00):
            today = date.today()
            weekback = date.today() - timedelta(days = 7)
            
            con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
            print("Database opened successfully")  
            cur = con.cursor()  
            cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(weekback)+"' and rtu_point_id ='13008'")
            yrows = cur.fetchall()
            print(len(yrows))
            dfcyesterday = pd.DataFrame(columns=['date','rtudata'])
            l=1
            for row in yrows:
                dfcyesterday.loc[l, 'date'] = str(row[0])
                dfcyesterday.loc[l, 'rtudata'] = str(row[1])
                l=l+1            
            mytime = datetime.now().time()        
            todaystr = today.strftime('%Y-%m-%d')        
            anlist2 =  pd.Series.tolist(dfcyesterday['rtudata'])
            anlist2 = ast.literal_eval(anlist2[0])
            timelist2 = []                
            anvalue2 =[]
            for x in anlist2:
                anvalue2.append(x['recordValue'])
                string = x['recordTime']
                string = todaystr + " " + string
                timelist2.append(string)
            yesdf = pd.DataFrame()        
            yesdf['timelist'] = timelist2
            yesdf['value'] = anvalue2
            
            filepath = "thursday/text-gen-best.hdf5"           
            
            
            model.load_weights(filepath)
            
            x_input = yesdf[-30:].values
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=1)
           
            for m in yhat[0]:
                plist.append(m)
        
    #at 1:00 
        elif(t1.hour == 00 and t1.minute == 15):
            
            
            cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(today)+"' and rtu_point_id ='13008'")
            rowsc = cur.fetchall()
            print(len(rowsc))
            rtudatasetc=[]
            dfc = pd.DataFrame(columns=['date','rtudata'])
            j=1
            for row in rowsc:
                dfc.loc[j, 'date'] = str(row[0])
                dfc.loc[j, 'rtudata'] = str(row[1])
                j=j+1
            
            anlist =  pd.Series.tolist(dfc['rtudata'])
            anlist = ast.literal_eval(anlist[0])
            timelist = []                
            anvalue =[]
            for x in anlist:
                anvalue.append(x['recordValue'])
                string = x['recordTime']
                string = todaystr + " " + string
                timelist.append(string)
            realdf = pd.DataFrame()        
            realdf['cvalue'] = anvalue
            realdf['time'] = timelist
            andf = realdf[:-30]
                
            x_input = andf[-30:].values
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat1 = model.predict(x_input, verbose=1)
            for m in yhat1[0]:
                plist.append(m)
    
    
    #from 2
                
        else:
            plist = []
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))    
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')

        
    
           
        while(True):
            t1 = datetime.now()
            if(t1.minute == 30 and t1.second == 00):
                today = date.today()
                todaystr = today.strftime('%Y-%m-%d')
                con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
                print("Database opened successfully")  
                cur = con.cursor()
                cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(today)+"' and rtu_point_id ='13008'")
                rowsc = cur.fetchall()
                print(len(rowsc))
                rtudatasetc=[]
                dfc = pd.DataFrame(columns=['date','rtudata'])
                j=1
                for row in rowsc:
                    dfc.loc[j, 'date'] = str(row[0])
                    dfc.loc[j, 'rtudata'] = str(row[1])
                    j=j+1
            
                anlist =  pd.Series.tolist(dfc['rtudata'])
                anlist = ast.literal_eval(anlist[0])
                timelist = []                
                anvalue =[]
                for x in anlist:
                    anvalue.append(x['recordValue'])
                    string = x['recordTime']
                    string = todaystr + " " + string
                    timelist.append(string)
                    realdf = pd.DataFrame()        
                    realdf['cvalue'] = anvalue
                    realdf['time'] = timelist
                realdf = realdf.set_index('time')
                raw_seq = realdf.values
            #    choose a number of time steps
                n_steps_in, n_steps_out = 30, 15
# split into samples

                X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

                n_features = 1
                X = X.reshape((X.shape[0], X.shape[1], n_features))
                y = y.reshape(y.shape[0],y.shape[1])
            
                model.fit(X,y,epochs = 10,verbose = 1)
            
                x_input = realdf[-30:].values
                x_input = x_input.reshape((1, n_steps_in, n_features))
                yhat1 = model.predict(x_input, verbose=1)
                for m in yhat1[0]:
                    plist.append(m)                   

                realdf.reset_index(level=0, inplace=True) 
                anlist = realdf.iloc[-15:,1]
                anlist = pd.Series.tolist(anlist)
                for x in plist[-15:]:
                   anlist.append(x)
                x_input = pd.Series(anlist[-30:]).values
                x_input = x_input.reshape((1, n_steps_in, n_features))
                yhat1 = model.predict(x_input, verbose=1)
                for m in yhat1[0]:
                    plist.append(m)
                for x in range(2):
                    x_input = pd.Series(plist[-30:]).values
                    x_input = x_input.reshape((1, n_steps_in, n_features))
                    yhat1 = model.predict(x_input, verbose=1)
                    for m in yhat1[0]:
                        plist.append(m)
                    
