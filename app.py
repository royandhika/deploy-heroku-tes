#!/usr/bin/python3

from flask import Flask
from flask import render_template
from flask import request

import os
import pickle

import pandas as pd
import numpy as np

with open('modelDecisionTreeClassifier.pkl','rb') as file:
    model = pickle.load(file)
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():   
    df = pd.DataFrame({'User ID':request.form.get('user_id'),
                       'Gender':request.form.get('user_gender'),
                       'Age':request.form.get('user_age'),
                       'EstimatedSalary':request.form.get('user_salary')}, index=[0])

    df = df.drop(['User ID'], axis=1)
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    prediction = model.predict(df)[0]
    
    if prediction == 0:
        predict = 'Tidak akan Membeli'
    else:
        predict = 'Akan Membeli'
    
    return render_template('result.html', result=predict)
    
if __name__ == "__main__":
    app.run()