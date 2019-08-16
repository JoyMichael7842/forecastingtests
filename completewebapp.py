# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:41:40 2019

@author: HP
"""

import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


df = realdf


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
       dash_table.DataTable(
       id='table',
       columns=[{"name": i, "id": i} for i in df.columns],
       style_table={'width': '200px'},
       data=df.to_dict('records'),
       fixed_rows={ 'headers': True, 'data': 0 },
       style_cell={'width': '100px'},
        ),
    ], style={'width': '20%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'line', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'line', 'name': u'Montréal'},
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )
    ], style= {'width': '0%', 'display': 'inline-block'})
])

if __name__ == '__main__':
    app.run_server()