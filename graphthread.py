# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:53:22 2019

@author: HP
"""

import psycopg2
import ast
from datetime import datetime,timedelta
import pandas as pd

t1 = datetime.now()- timedelta(days=1)
today = t1.today()-timedelta(days=0)
yesterday = today - timedelta(days=1) 
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

todaystr = today.strftime('%Y-%m-%d')       
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
yesdf.timelist = pd.to_datetime(yesdf.timelist,format = '%Y-%m-%d %H:%M')
yesdf = yesdf.set_index('timelist')

        

cur.execute("select RTU_POINT_RECORD from sldcsch.rtu_point_real_fcast_data where RTU_DATA_DATE='"+str(today)+"' and rtu_point_id='13008'")
rfrows = cur.fetchall()
predlist = rfrows.copy()
predlist1 = ast.literal_eval(predlist[0][0])
timelist3 = []                
anvalue3 =[]
for x in predlist1:
    anvalue3.append(x['recordValue'])
    string = x['recordTime']
    string = todaystr + " " + string
    timelist3.append(string)
predf = pd.DataFrame()        
predf['timelist'] = timelist3
predf['value'] = anvalue3
predf.timelist = pd.to_datetime(predf.timelist,format = '%Y-%m-%d %H:%M')
predf = predf.set_index('timelist')
    


ax = realdf.plot(figsize=(20,10))
predf.plot(ax = ax)
yesdf.plot(ax=ax)

realdf.reset_index(level=0, inplace=True)
predf.reset_index(level=0, inplace=True)
yesdf.reset_index(level=0,inplace = True)


anpdf1 = realdf.iloc[7:720:30,:]
anpdf2 = realdf.iloc[15:720:30,:]
anpdf3 = realdf.iloc[22:720:30,:]
anpdf4 = realdf.iloc[29:720:30,:]

anpdf5 = predf.iloc[7:720:30,:]
anpdf6 = predf.iloc[15:720:30,:]
anpdf7 = predf.iloc[22:720:30,:]
anpdf8 = predf.iloc[29:720:30,:]

anpdf9 = yesdf.iloc[7:720:30,:]
anpdf10 = yesdf.iloc[15:720:30,:]
anpdf11 = yesdf.iloc[22:720:30,:]
anpdf12 = yesdf.iloc[29:720:30,:]

hrlist = anpdf1.append(anpdf2,ignore_index=True).append(anpdf3,ignore_index=True).append(anpdf4,ignore_index=True).sort_values("time")
hplist = anpdf5.append(anpdf6).append(anpdf7).append(anpdf8).sort_values("timelist")
hylist = anpdf9.append(anpdf10).append(anpdf11).append(anpdf12).sort_values("timelist") 

andf = pd.merge(hylist,pd.merge(hrlist, hplist, left_on='time', right_on='timelist', how='right'),how = 'right',on = 'timelist')

andf = pd.merge(pd.merge(hrlist, hplist, left_on='time', right_on='timelist', how='right'),hylist,left_on='time', right_on='timelist', how='right').drop('time', axis=1)
andf['error_pred'] = ((andf['cvalue'] - andf['value_x'])/andf['cvalue'])*100
andf['error_yes'] = ((andf['cvalue'] - andf['value_y'])/andf['cvalue'])*100
andf.error_pred = andf.error_pred.round(2)
andf.error_yes = andf.error_yes.round(2)
andf = andf.sort_index()

andf = andf.drop('timelist',axis=1)

