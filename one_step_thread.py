# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:12:42 2019

@author: HP
"""
from numpy import array
import pandas as pd
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import date,datetime,timedelta
import psycopg2
import json
import ast
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
tempweekday=-1
tempday=-1
t1 = datetime.now()
weekd = t1.weekday()
#model retrieve 
con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
print("Database opened successfully")  
cur = con.cursor()  
while(true):
 
    
    if(tempday!=today):
 
        cur.execute("select model_json,weights,trained_for_day,max(rec_inserted_time) from sldcsch.fcast_model_weights where trained_for_day="+str(weekd)+"group by model_json,weights,trained_for_day")
        rows = cur.fetchall()
        print(len(rows))
        
        df = pd.DataFrame(columns=['weekday','modeljson','weights'])
        i=1
        for row in rows:
            df.loc[i, 'modeljson'] = str(row[0])
            df.loc[i, 'weights'] = str(row[1])
            df.loc[i,'weekday'] = str(row[2])
            i=i+1
            
        print(df)
    
        json_string_2 = df.iloc[0,1]
        model = model_from_json(json_string_2)
        
        json_string = df.iloc[0,2]
        datastore = json.loads(json_string)
        anlist2 = []
        for i,weights in enumerate(datastore):
            dataaa = np.array(weights)
            anlist2.append(dataaa)    
        
        anl = []
        for j in range(len(anlist2)):
            an = np.array([np.float32(i) for i in anlist2[j]])
            anl.append(an)
        model.set_weights(anl)
    
        tempday=today
 
     
 
    if(t1.hour == 00 and t1.minute == 00 and t1.second == 00):
    
       
    
        yesterday = t1.today() - timedelta(days = 1)    
        print("Database opened successfully")  
        cur = con.cursor()  
        cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(yesterday)+"' and rtu_point_id ='13008'")
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
        yesdf = yesdf.set_index('timelist')
        
        x_input = yesdf[-60:].values
        dataset_total = x_input.copy() 
        n_future_preds = 30
        preds_moving = []
        moving_test_window = dataset_total[-60:]
        moving_test_window = sc.fit_transform(moving_test_window)
        moving_test_window = np.array(moving_test_window)
        moving_test_window = moving_test_window.reshape(1,60,1)
        for i in range(n_future_preds):
            preds_one_step = model.predict(moving_test_window)
            preds_moving.append(preds_one_step[0,0])
            preds_one_step = preds_one_step.reshape(1,1,1)
            moving_test_window = np.concatenate((moving_test_window[:,1:,:],preds_one_step),axis = 1)
        
        myarr = np.array(preds_moving)
        myarr = myarr.reshape(-1,1)
        ans = sc.inverse_transform(myarr)
        
    #predicted_stock_price[0][0] = dataset_test[0][0]
        
        dts = [dt.strftime('%H:%M') for dt in 
               datetime_range(t1,t1+timedelta(minutes = 60) ,timedelta(minutes=2))]
        
        print(dts)
        
        andf = mydf[4119:4149].copy()
        pdf = pd.DataFrame()
        pdf['time'] = andf['timelist']
        pdf['value'] = ans
        pdf = pdf.set_index('time')