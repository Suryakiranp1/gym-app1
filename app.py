# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:34:50 2021

@author: surya
"""

import flask
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('home.html')



@app.route('/predict', methods = ['Post'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('back.html', data=pred)

if __name__ == '__main__':
    app.run(debug=True)
    
    