# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:06:46 2019

@author: HP
"""

import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


df = andf
list1 = hrlist.cvalue.tolist()
list2 = hplist.value.tolist()
list3 = hylist.value.tolist()
timelist1 = hrlist.time.tolist() 
timelist2 = hplist.timelist.tolist() 
timelist3 = hylist.timelist.tolist() 
datap = [{'Name':'abc','val' : 123},{'Name' : 'xyz','val' : 345}]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Div([
       dash_table.DataTable(
       id='table',
       columns=[{"name": i, "id": i} for i in df.columns],
       style_table={'width': '800px'},
       data=df.to_dict('records'),
       fixed_rows={ 'headers': True, 'data': 1 },
       style_cell={
        # all three widths are needed
        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],
        ),
               
    ], style={'width': '70%','display': 'inline-block'}),
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
        dash_table.DataTable(
            id = 'table2',
            columns=[{"name": 'Name', "id": 'Name'},{"name": 'val', "id": 'val'}],
            style_table={'width': '150px'},
            data=datap,
            fixed_rows={ 'headers': True, 'data': 0 },
            style_cell={'width': '100px','display': 'inline-block'},
        ),
        
    ], style= {'width': '49%'})
])
# 'display': 'inline-block'

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port = 8080)
    
    
"""       style_cell={
        # all three widths are needed
        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],"""

    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],
