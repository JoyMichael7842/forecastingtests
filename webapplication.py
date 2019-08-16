# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:56:53 2019

@author: HP
"""

import dash
import dash_table
import pandas as pd


df = realdf


app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    style_table={'width': '200px'},
    data=df.to_dict('records'),
    fixed_rows={ 'headers': True, 'data': 0 },
    style_cell={'width': '100px'}
    
    

)

if __name__ == '__main__':
    app.run_server()