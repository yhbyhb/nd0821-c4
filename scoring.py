from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 
testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))

#################Function for model scoring
def score_model(model_path, testdata):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    model_file_path = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

    X_test = testdata[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y_test = testdata['exited'].values.reshape(-1, 1)

    y_pred = model.predict(X_test)

    f1score = metrics.f1_score(y_test, y_pred)

    latest_score_file = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    with open(latest_score_file, 'w') as score_file:
        score_file.write(str(f1score))

    return f1score

if __name__ == '__main__':
    score_model(model_path, testdata)