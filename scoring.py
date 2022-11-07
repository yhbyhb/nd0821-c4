import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json
import utils


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 
testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))

#################Function for model scoring
def score_model(model_path, test_data_frame):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    model = utils.load_model(model_path)

    X_test, y_test = utils.split_data(test_data_frame)

    y_pred = model.predict(X_test)

    f1score = metrics.f1_score(list(y_test), list(y_pred))

    latest_score_file = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    with open(latest_score_file, 'w') as score_file:
        score_file.write(str(f1score))

    return f1score

if __name__ == '__main__':
    score_model(model_path, testdata)