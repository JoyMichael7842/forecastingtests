# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:36:42 2019

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
        dates_list.append(stringDate)#print(dates_list)
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


mydf.reset_index(level=0, inplace=True)
mydf = mydf[-6*720:]
indexedDataset = mydf.copy()
indexedDataset = indexedDataset.set_index('timelist')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10,6

from datetime import datetime
indexedDataset.head(5)

plt.xlabel("time")
plt.ylabel("values")
plt.plot(indexedDataset.values)

rolmean = indexedDataset.rolling(window = 720).mean()

rolstd = indexedDataset.rolling(window = 720).std()

plt.figure(figsize = (20,10))
orig = plt.plot(indexedDataset.values,color = 'blue',label = 'original')
mean = plt.plot(rolmean.values,color = 'red',label = 'Rolling Mean')
std = plt.plot(rolstd.values,color = 'black',label = 'Rolling std')
plt.legend(loc='best')
plt.title('rolling mean and std deviation')
plt.show(block = False)

from statsmodels.tsa.stattools import  adfuller

print('Results of dickey-fuller test:')
dftest = adfuller(indexedDataset['value'],autolag = 'AIC')
                                 
dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
print(dfoutput)     

indexedDataset_logscale = np.log(indexedDataset)
plt.plot(indexedDataset_logscale.values)

movingAverage = indexedDataset_logscale.rolling(window = 720).mean()
movingSTD = indexedDataset_logscale.rolling(window = 720).std()
plt.plot(indexedDataset_logscale.values)
plt.plot(movingAverage.values,color = 'red')

datasetLogScaleMinusMovingAverage = indexedDataset_logscale-movingAverage
datasetLogScaleMinusMovingAverage.head(720)

datasetLogScaleMinusMovingAverage.dropna(inplace = True)
datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import  adfuller
def test_stationary(timeseries):
    #Determinig rolling statistics
    movingAverage = timeseries.rolling(window = 720).mean()
    movingstd = timeseries.rolling(window = 720).std()
    
    #plot rolling statistics:
    plt.figure(figsize = (20,10))
    orig = plt.plot(timeseries.values,color = 'blue',label = 'original')
    mean = plt.plot(movingAverage.values,color = 'red',label = 'Rolling Mean')
    std = plt.plot(movingstd.values,color = 'black',label = 'Rolling std')
    plt.legend(loc='best')
    plt.title('rolling mean and std deviation')
    plt.show(block = False)

    #dickey-fuller test 
    print('Results of dickey-fuller test:')
    dftest = adfuller(timeseries['value'],autolag = 'AIC')                                 
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key] = value
    print(dfoutput)

test_stationary(datasetLogScaleMinusMovingAverage)   

exponentialDecayweightedaverage =   indexedDataset_logscale.ewm(halflife = 720,min_periods = 0,adjust = True).mean()
plt.plot(indexedDataset_logscale.values)
plt.plot(exponentialDecayweightedaverage.values,color = 'red')

datasetlogscaleMinusMovingExponentialDecayAverage = indexedDataset_logscale - exponentialDecayweightedaverage
test_stationary(datasetlogscaleMinusMovingExponentialDecayAverage)

datasetLogDiffShifting = indexedDataset_logscale - indexedDataset_logscale.shift()
plt.plot(datasetLogDiffShifting.values)

datasetLogDiffShifting.dropna(inplace = True)
test_stationary(datasetLogDiffShifting)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logscale,freq = 720)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid 

plt.subplot(411)
plt.plot(indexedDataset_logscale.values,label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend.values,label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal.values,label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual.values,label = 'Residuals')
plt.plot(loc = 'best')
plt.tight_layout()

decomposedlogdata = residual
decomposedlogdata.dropna(inplace = True)
test_stationary(decomposedlogdata)  

decomposedLogData = residual
decomposedLogData.dropna(inplace = True)
test_stationary(decomposedLogData) 

from statsmodels.tsa.stattools import acf,pacf

lag_acf = acf(datasetLogDiffShifting,nlags= 20)
lag_pacf = pacf(datasetLogDiffShifting,nlags= 20,method = 'ols')

#plot Acf
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color = 'gray')
plt.title('Autocorrelation Function')
#plot pacf

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color = 'gray')
plt.title('Autocorrelation Function')

from statsmodels.tsa.arima_model import ARIMA

#Ar model
model = ARIMA(indexedDataset_logscale,order=(2,1,2))
results_AR = model.fit(disp=-1)
plt.figure(figsize=(20,10))
plt.plot(datasetLogDiffShifting.values)
plt.plot(results_AR.fittedvalues.values,color = 'red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting["value"])**2))
print("plotting ar model")

#MA model
model = ARIMA(indexedDataset_logscale,order=(2,1,1))
results_MA = model.fit(disp = -1)
plt.plot(datasetLogDiffShifting.values)
plt.plot(results_MA.fittedvalues.values,color = 'red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["value"])**2))
print("plotting ar model")

model = ARIMA(indexedDataset_logscale,order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting.values)
plt.plot(results_ARIMA.fittedvalues.values,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["value"])**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy = True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(indexedDataset_logscale["value"].ix[0],index=indexedDataset_logscale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset.values)
plt.plot(predictions_ARIMA.values)

indexedDataset_logscale
results_ARIMA.plot_predict(1,5100)
x = results_ARIMA.forecast(steps = 720)

x_cumsum = x[1].cumsum()
print(predictions_ARIMA_diff_cumsum.head())


forecast = np.exp(x[1])

plt.plot(forecast.values)
