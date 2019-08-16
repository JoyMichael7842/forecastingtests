# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:14:02 2019

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
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import date,datetime,timedelta
import psycopg2
import json
import ast

today = date.today() +timedelta(days=0)
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


mydf.reset_index(level=0, inplace=True)
indexedDataset = mydf.copy()
indexedDataset = indexedDataset.set_index('timelist')

# Importing the training set
training_set = indexedDataset.values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

weights_list = regressor.get_weights()

modeljson = regressor.to_json()
modeljson = modeljson.replace('\'','\"')
anlist = []
for i,weights in enumerate(weights_list):
    dataaa = np.ndarray.tolist(weights)
    anlist.append(dataaa)

weightsjson = json.dumps(anlist)
"""
con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
print("Database opened successfully")  
cur = con.cursor()  
cur.execute("insert into sldcsch.fcast_model_weights(rtu_point_id,model_json,weights,trained_from_date,trained_to_date,trained_for_day,training_execute_date) values('13008','"+modeljson+"','"+weightsjson+"','"+dates_list[0]+"','"+dates_list[-2]+"','"+str(weekd)+"','2019-06-26')")
con.commit()

"""