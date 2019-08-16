# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:22:56 2019

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
dayfirst = startdate + timedelta(days=-7)
#print ("Start Date",dayfirst.strftime("%Y-%m-%d"))

print ("Data between Start Date",dayfirst.strftime("%Y-%m-%d"),"End Date",previousdayf)

#delta=previousday-dayfirst
delta=previousday-dayfirst

dates_list = []

for i in range(delta.days + 1):
    mydate=dayfirst + timedelta(i)
    weekdayvalue=mydate.weekday()   
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

mydf = mydf[:5128]
import fbprophet

andf = pd.DataFrame()
andf['ds'] = mydf['timelist'].copy()
andf['y'] = mydf['value'].copy()
#andf = andf.set_index('ds')

m = fbprophet.Prophet()
m.fit(andf)

future = m.make_future_dataframe(periods = 60,freq = 'min')
forecast = m.predict(future)

m.plot(forecast)

m.plot_components(forecast)

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
t1 = datetime.now()
todaystr = t1.today().strftime('%Y-%m-%d')       
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
realdf.time = pd.to_datetime(realdf.time,format = '%Y-%m-%d %H:%M')
realdf = realdf.set_index('time')

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%Y-%m-%d T%H:%M Z') for dt in 
       datetime_range(datetime(2019, 9, 1, 7), datetime(2016, 9, 1, 9+12), 
       timedelta(minutes=15))]


predtime = forecast.iloc[-60:,0]
predval = forecast.iloc[-60:,-1]
predf = pd.DataFrame()
predf['time'] = mydf.iloc[5127:5187,0]
predf['val'] = predval
predf = predf.set_index('time')

ax = realdf.plot()
predf.plot(ax=ax)
predf.plot()