# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:21:15 2019

@author: HP
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
import time
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
import psycopg2
import json
import ast
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

tempday=-1
t1 = datetime.now()
weekd = t1.weekday()
anlist = []
anlist2 = []
sc = MinMaxScaler(feature_range = (0, 1))
tempweekday=-1
tempday=-1
anindex = 60
t1 = datetime.now()
date_time_str = '2018-06-29 00:00:00'  
t2 = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
minlist = [dt.strftime('%H:%M') for dt in 
               datetime_range(t2,t2+timedelta(days = 1) ,timedelta(minutes=2))]
#model retrieve 
con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
print("Database opened successfully")  
cur = con.cursor() 
while(True):
    if(tempday != weekd):
        print('changing model')
        totlist = []
        cur.execute("select model_json,weights,trained_for_day,max(rec_inserted_time) from sldcsch.fcast_model_weights where trained_for_day="+str(weekd)+"group by model_json,weights,trained_for_day")
        rows = cur.fetchall()
        
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
        tempday = weekd
        
        
    t1 = datetime.now()
    weekd = t1.weekday()
    if  t1.hour == 00 and t1.minute == 00 and t1.second == 00:
        print("&&&&&&&&&&&&&&&&& Zerio Condition inside")
        yesterday = t1.today() - timedelta(days = 1)    
        #print("Database opened successfully")  
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
        todaystr = t1.today().strftime('%Y-%m-%d')        
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
        
        prediction = []
        for i in range(len(ans)):
            predictarray = {}
            predictarray['recordTime'] = minlist[i]
            predictarray['recordValue'] = ans[i][0]
            prediction.append(predictarray)
        print('prediction at 00:'+str(prediction))
        for x in prediction:
            totlist.append(x)
        
        
        
        today = t1.today()
        cur.execute("select RTU_POINT_RECORD from sldcsch.rtu_point_real_fcast_data where RTU_DATA_DATE='"+str(today)+"' and rtu_point_id='13008'")
        rfrows = cur.fetchall()
        if len(rfrows) > 0:
            cur.execute("update sldcsch.rtu_point_real_fcast_data set rtu_point_record='"+str(totlist).replace('\'','\"')+"' where rtu_point_id='13008' and rtu_data_date='"+str(today)+"'")
            print("updated record")
        else:
            cur.execute("insert into sldcsch.rtu_point_real_fcast_data (rtu_point_id,rtu_data_date,rtu_point_record) values('13008','"+str(today)+"','"+str(totlist).replace('\'','\"')+"')")
            print("inserted record")
        con.commit()
        time.sleep(5)
    
    #if t1.hour == 1 and t1.minute == 00 and t1.second == 00:
        
        
    if  t1.hour!=1 and t1.hour!=00 and t1.minute == 00 and t1.second == 00:
        print("################ Not One Hour Condition inside")
        today = t1.today()    
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
        realdf = realdf.set_index('time')
        
        training_set = realdf.values
        training_set_scaled = sc.fit_transform(training_set)
        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(X_train, y_train, epochs = 30, batch_size = 32)
        
        x_input = realdf[-60:].values
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
        
        t3 = datetime.now() 
        date_time_str1 = '2018-06-29 '+str(t3.hour)+':00:00'  
        t4 = datetime.strptime(date_time_str1, '%Y-%m-%d %H:%M:%S')
        minlist2 = [dt.strftime('%H:%M') for dt in 
           datetime_range(t4,t4+timedelta(hours = 1) ,timedelta(minutes=2))]
    
        
        prediction = []
        for i in range(len(ans)):
            predictarray = {}
            predictarray['recordTime'] = minlist2[i]
            predictarray['recordValue'] = ans[i][0]
            prediction.append(predictarray)
            
        print('prediction at '+str(t3.hour)+':'+str(prediction))
        
    
        
        cur.execute("select RTU_POINT_RECORD from sldcsch.rtu_point_real_fcast_data where RTU_DATA_DATE='"+str(today)+"' and rtu_point_id='13008'")
        rfrows = cur.fetchall()
        
        
        existdatalist=[]
        for row in rfrows:
            existdatalist.append(str(row[0]))
        existdatalist = ast.literal_eval(existdatalist[0])
       
        print(existdatalist)
        for x in prediction:
            existdatalist.append(x)
        
        
        if len(rfrows) > 0:
            cur.execute("update sldcsch.rtu_point_real_fcast_data set rtu_point_record='"+str(existdatalist).replace('\'','\"')+"' where rtu_point_id='13008' and rtu_data_date='"+str(today)+"'")
            print("updated record")
        else:
            cur.execute("insert into sldcsch.rtu_point_real_fcast_data (rtu_point_id,rtu_data_date,rtu_point_record) values('13008','"+str(today)+"','"+str(existdatalist).replace('\'','\"')+"')")
            print("inserted record")
        con.commit()
        time.sleep(30*60)
    else:
        print("******************** else Condition inside")
        today = t1.today()
        cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(today)+"' and rtu_point_id ='13008'")
        rowsc = cur.fetchall()
        #print(len(rowsc))
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
        todaystr = t1.today().strftime('%Y-%m-%d')        
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
        
        x_input = [np.append(yesdf[-30:].values,realdf[:30].values)]
        x_input = np.array(x_input)
        x_input =x_input.T
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
        
        prediction = []
        index = 30
        for i in range(len(ans)):
            predictarray = {}
            predictarray['recordTime'] = minlist[index]
            predictarray['recordValue'] = ans[i][0]
            prediction.append(predictarray)
            index = index+1
        print('prediction at 1:'+str(prediction))       
        
        
        cur.execute("select RTU_POINT_RECORD from sldcsch.rtu_point_real_fcast_data where RTU_DATA_DATE='"+str(today)+"' and rtu_point_id='13008'")
        rfrows = cur.fetchall()
        existdatalist=[]
        for row in rfrows:
            existdatalist.append(str(row[0]))
        existdatalist = ast.literal_eval(existdatalist[0])
        print(existdatalist)
        for x in prediction:
            existdatalist.append(x)
        if len(rfrows) > 0:
            cur.execute("update sldcsch.rtu_point_real_fcast_data set rtu_point_record='"+str(existdatalist).replace('\'','\"')+"' where rtu_point_id='13008' and rtu_data_date='"+str(today)+"'")
            print("updated record")
        else:
            cur.execute("insert into sldcsch.rtu_point_real_fcast_data (rtu_point_id,rtu_data_date,rtu_point_record) values('13008','"+str(today)+"','"+str(existdatalist).replace('\'','\"')+"')")
            print("inserted record")
        con.commit()    
        
        time.sleep(5)
        