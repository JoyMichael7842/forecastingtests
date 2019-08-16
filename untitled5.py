# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:42:48 2019

@author: HP
"""
yesterday = today -timedelta(days=1)
con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
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

sind = yesdf.loc[yesdf.timelist == realdf.iloc[-1,1]].index
if len(sind) == 0:
    print('no data')
else:
    sind = sind[0]
    eind = sind-120

    pardf  = yesdf.iloc[144:234,:]
    
antime = realdf.iloc[450:540,1]
pldf = pd.DataFrame()
pldf['time'] = antime
pldf['values'] = plist[:90]

pardf = pardf.set_index('timelist')
pldf = pldf.set_index('time')
realdf = realdf.set_index('time')

ax =realdf.plot()
pldf.plot(ax=ax)
pardf.plot(ax=ax)

plt.plot(realdf)
plt.plot(pldf)
plt.plot(pardf)
plt.show()