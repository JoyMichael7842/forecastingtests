# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:47:30 2019

@author: HP
"""

from numpy import array

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

import pandas as pd
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

today = date.today() +timedelta(days=3)
#today = date.today()-timedelta(days=4)
print("Today",today)
ddttt=pd.Timestamp(today)
ddtttf=ddttt.strftime("%Y-%m-%d")

weekd=today.weekday() #Week Day of Next Day
print("Weekday",weekd)
#Previous Day for getting list of Dates for fetching training data
previousday = ddttt + timedelta(days=-1)
previousdayf=previousday.strftime("%Y-%m-%d")
print ("Previous Day",previousdayf)
#Start Date for the Data Samples
startdate=pd.Timestamp(previousdayf)
dayfirst = startdate + timedelta(days=-366)
#print ("Start Date",dayfirst.strftime("%Y-%m-%d"))

print ("Data between Start Date",dayfirst.strftime("%Y-%m-%d"),"End Date",previousdayf)

#delta=previousday-dayfirst
delta=previousday-dayfirst

dates_list = []

for i in range(delta.days + 1):
    mydate=dayfirst + timedelta(i)
    weekdayvalue=mydate.weekday()
   
    if(weekdayvalue==weekd ):
        stringDate=mydate.strftime("%Y-%m-%d")
        dates_list.append(stringDate)
#print(dates_list)
dates_list.append(previousdayf)    
dates_list.append(ddtttf)    
print("Total Dates",len(dates_list))
ss=''
for i in dates_list:
    ss=ss+"'"+i+"',"
strql=ss[:len(ss)-1]

print(strql)
#time slots


timelist=[]
ddttt=pd.Timestamp(today)
#for ii in range(0,1440):
intvalue=0
for ii in range(0,720):
    nine_hours_from_now = ddttt +timedelta(minutes=intvalue)
    timelist.append(nine_hours_from_now.strftime("%H:%M"))
    intvalue=intvalue+2

print(timelist)
#time stamp

con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
print("Database opened successfully")  
cur = con.cursor()  
cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date in ("+strql+") and rtu_point_id ='13008'")
rows = cur.fetchall()
print(len(rows))
#rtudataset=[]
df = pd.DataFrame(columns=['date','rtudata'])
i=1
for row in rows:
    df.loc[i, 'date'] = str(row[0])
    df.loc[i, 'rtudata'] = str(row[1])
    i=i+1
    
df.date = pd.to_datetime(df.date, format='%Y-%m-%d') 
df = df.sort_values(by=['date'])
df.date = df.date.dt.strftime('%Y-%m-%d')

print(df)



previousdaydataset=[]
dataarray=[]
totaldf = pd.DataFrame()

for index, row in df.iterrows():
    recorddate=row['date']
   # print(recorddate)
    #valuedddd=row['rtudata']
    datastore = json.loads(str(row['rtudata']))
    #print(recorddate,"data--->",datastore)
    
    it=0
    tempid=0
    print(recorddate,previousdayf)
    if recorddate==previousdayf:
        previousdaydataset.append(datastore)
        print("previous day",recorddate)
    else:
        for timettt in timelist:
            ittemp=0
            tempmw=0
            datavalues={}
            #print(timettt)
            for timemm in datastore:
                if timemm['recordTime']==timettt:
                    datavalues['recordTime']=recorddate+' '+timemm['recordTime']
                    tempmw=timemm['recordValue']
                    datavalues['recordValue']=tempmw
                    dataarray.append(datavalues)
                    tempid=1
                    #print("I am inside")
                    ittemp=1
            if ittemp==1:
                it=it+1
        


print(dataarray)

mydf = pd.DataFrame()
timelist = []
vallist = []
for  x in dataarray:
    if(x['recordValue']>12000):
        timelist.append(x['recordTime'])
        vallist.append(7958.89)
    else:
        timelist.append(x['recordTime'])
        vallist.append(x['recordValue'])
    
mydf['timelist'] = timelist
mydf['value'] = vallist

mydf.timelist = pd.to_datetime(mydf.timelist,format = '%Y-%m-%d %H:%M')

mydf = mydf.set_index('timelist')

plt.figure(figsize = (20,6))
plt.plot(mydf.values)
plt.title('dataframe')
plt.grid(True)
plt.show()

#get the time HH:MM for start the program for trigger every 15 minutes
# 00:15 60 samples from previous day one hour 0..24


vallist2 = []
tlist = []


# define input sequence
raw_seq = mydf.values
# choose a number of time steps
n_steps_in, n_steps_out = 30, 15
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
#for i in range(len(X)):
#	print(X[1], y[1])
    
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape(y.shape[0],y.shape[1])

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# checkpoint
filepath = "sundayweights/text-gen-best.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
model.fit(X, y, epochs=100, verbose=1,callbacks=[checkpoint])
