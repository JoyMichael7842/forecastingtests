# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:08:18 2019

@author: HP
"""
    
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
import psycopg2
import ast
from datetime import datetime,timedelta
import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_html_components as html

def myfun(mdate):
    
    t1 = datetime.now()
    today = mdate
    yesterday = today - timedelta(days=1) 
    con = psycopg2.connect(database="SLDCREP", user="postgres", password="createdon16082018", host="192.168.168.125", port="5432")
    print("Database opened successfully")  
    cur = con.cursor() 
    cur.execute("SELECT rtu_data_date, rtu_point_record from sldcsch.rtu_point_data where rtu_data_date='"+str(today)+"' and rtu_point_id ='13008'")
    rowsc = cur.fetchall()
    print(len(rowsc))
    
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
    hplist = anpdf5.append(anpdf6,ignore_index=True).append(anpdf7,ignore_index=True).append(anpdf8,ignore_index=True).sort_values("timelist")
    hylist = anpdf9.append(anpdf10,ignore_index=True).append(anpdf11,ignore_index=True).append(anpdf12,ignore_index=True).sort_values("timelist") 
    
    andf = pd.merge(hylist,pd.merge(hrlist, hplist, left_on='time', right_on='timelist', how='right'),how = 'right',on = 'timelist')
    
    
    andf = andf.drop('time',axis=1) 
    
    ldf = pd.DataFrame()
    ldf['time'] = andf['timelist']
    ldf['realtime'] = andf['cvalue']
    ldf['prediction'] = andf['value_y']
    ldf['yesterday'] = andf['value_x']
    ldf['deviation'] = ldf['realtime']-ldf['prediction']
    ldf['error%'] = (ldf['deviation']/ldf['realtime']) *100
    ldf.deviation = ldf.deviation.round(2)
    ldf['error%'] = ldf['error%'].round(2)
    ldf.time = ldf.time.dt.strftime('%H:%M')
       
    
    
    
    df = ldf
    list1 = hrlist.cvalue.tolist()
    list2 = hplist.value.tolist()
    list3 = hylist.value.tolist()
    timelist1 = hrlist.time.tolist() 
    timelist2 = hplist.timelist.tolist() 
    timelist3 = hylist.timelist.tolist() 
    datap = [{'time':hplist.iloc[-4,0].strftime("%H:%M"),'value' : hplist.iloc[-4,1]},{'time':hplist.iloc[-3,0].strftime("%H:%M"),'value' : hplist.iloc[-3,1]},{'time':hplist.iloc[-2,0].strftime("%H:%M"),'value' : hplist.iloc[-2,1]},{'time':hplist.iloc[-1,0].strftime("%H:%M"),'value' : hplist.iloc[-1,1]}]

    return df,datap,timelist1,timelist2,timelist3,list1,list2,list3

df,datap,timelist1,timelist2,timelist3,list1,list2,list3 = myfun(datetime.today())
t5 =  datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


"""
 html.Div([
    html.Div([
        html.Div([
            
            dash_table.DataTable(
       id='table',
       columns=[{"name": i, "id": i} for i in df.columns],
       style_table={'width': '500px','maxHeight': '300',},
       data=df.to_dict('records'),
       fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={
        # all three widths are needed
        'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',        
    },
       
        ),
        ], className="six columns"),

        html.Div([
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=datetime(1995, 8, 5),
        max_date_allowed=datetime(2017, 9, 19),
        initial_visible_month=datetime(2017, 8, 5),
        date=str(datetime(2017, 8, 25, 23, 59, 59))
    ),
    html.Div(id='output-container-date-picker-single'),
    dcc.Link('Go back to home', href='/')
]),
    ], className="row"),
    html.Div([
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': timelist1, 'y': list1, 'type': 'line', 'name': 'real'},
                        {'x': timelist2, 'y': list2, 'type': 'line', 'name': u'prediction'},
                        {'x': timelist3, 'y': list3, 'type': 'line', 'name': u'yesterday'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            ),
            
            ]),   
    html.Div([
    dcc.Link('Go to Page 1', href='/page-1')
    
])      
         
])
"""
# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.Div([
        html.Div([
            
            dash_table.DataTable(
       id='table',
       columns=[{"name": i, "id": i} for i in df.columns],
       style_table={'width': '500px','maxHeight': '300',},
       data=df.to_dict('records'),
       fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={
        # all three widths are needed
        'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',        
    },
       
        ),
        ], className="six columns"),

        html.Div([
             html.H3('The time is: ' + t5),
             dash_table.DataTable(
            id = 'table2',
            columns=[{"name": 'Time', "id": 'time'},{"name": 'Value', "id": 'value'}],
            style_table={'width': '600px','maxHeight': '300'},
            data=datap,
            fixed_rows={ 'headers': True, 'data': 0 },
            style_cell={'width': '300px','height': '80px', 'font_size': '25px','text_align': 'center','display': 'inline-block'},
        ),
        ], className="six columns"),
    ], className="row"),
    html.Div([
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': timelist1, 'y': list1, 'type': 'line', 'name': 'real'},
                        {'x': timelist2, 'y': list2, 'type': 'line', 'name': u'prediction'},
                        {'x': timelist3, 'y': list3, 'type': 'line', 'name': u'yesterday'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            ),
            
            ]),   
    html.Div([
    dcc.Link('Go to Page 1', href='/page-1')
    
])      
         
])
page_1_layout = html.Div([
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=datetime(2019, 6, 5),
        max_date_allowed=datetime.today().date(),
        initial_visible_month=datetime.today().date(),
        date=str(datetime.today())
    ),
    html.Div(id='output-container-date-picker-single'),
    dcc.Link('Go back to home', href='/')
])


@app.callback(
    Output('output-container-date-picker-single', 'children'),
    [Input('my-date-picker-single', 'date')])
def update_output(date):
    
    if date is not None:
        pdate = datetime.strptime(date.split(' ')[0], '%Y-%m-%d')
        df,datap,timelist1,timelist2,timelist3,list1,list2,list3 = myfun(pdate)
        date_string = pdate.strftime('%B %d, %Y')
        
        return html.Div([
    html.Div([
        html.Div([
            
            dash_table.DataTable(
       id='table',
       columns=[{"name": i, "id": i} for i in df.columns],
       style_table={'width': '500px','maxHeight': '300',},
       data=df.to_dict('records'),
       fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={
        # all three widths are needed
        'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',        
    },
       
        ),
        ], className="six columns"),

        html.Div([
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=datetime(2019, 6, 5),
         max_date_allowed=datetime.today().date(),
        initial_visible_month=datetime.today().date(),
        date=str(datetime.today())
    ),
    html.Div(id='output-container-date-picker-single'),
    dcc.Link('Go back to home', href='/')
]),
    ], className="row"),
    html.Div([
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': timelist1, 'y': list1, 'type': 'line', 'name': 'real'},
                        {'x': timelist2, 'y': list2, 'type': 'line', 'name': u'prediction'},
                        {'x': timelist3, 'y': list3, 'type': 'line', 'name': u'yesterday'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            ),
            
            ]),   
    html.Div([
    dcc.Link('Go to Page 1', href='/page-1')
    
])      
         
])




# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port = 7000)