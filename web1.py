# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:04:01 2019

@author: HP
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello world"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
    