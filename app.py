from flask import Flask, session, jsonify, request
import pandas as pd
import pickle
import json
import os

import utils
from diagnostics import (
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list
)
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'])
prediction_model_path = os.path.join(os.getcwd(), model_path, "trainedmodel.pkl")
prediction_model = pickle.load(open(prediction_model_path, "rb"))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('inputdata')
    # print(filename)
    test_file_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(test_file_path) is False:
        return f"{test_file_path} doesn't exist"

    test_data_frame = pd.read_csv(test_file_path)
    X, _ = utils.split_data(test_data_frame)

    pred = prediction_model.predict(X)
    return str(pred), 200 #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    test_data_path = os.path.join(config['test_data_path'])
    testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    f1score = score_model(model_path, testdata)
    return jsonify(f1score), 200 #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    statistics = dataframe_summary()
    return jsonify(statistics), 200 #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    timing = execution_time()
    missing = missing_data()
    dependencies = outdated_packages_list()
    res = {
        'timing' : timing,
        'missing_data' : missing,
        'dependency_check' : dependencies,
    }
    return jsonify(res), 200#add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
