from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import {
    model_predictions,
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list
}
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = pickle.load(open(os.path.join(os.getcwd(), config['prod_deployment_path'], "trainedmodel.pkl"), "rb"))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('inputdata')
    # print(filename)
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(file_path) is False:
        return f"{filename} doesn't exist"

    pred = model_predictions(file_path)
    return str(pred) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1score = score_model()
    return str(f1score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    statistics = dataframe_summary()
    return str(statistics) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    timing = execution_time()
    missing_data = missing_data()
    dependencies = outdated_packages_list()
    res = {
        'timing' : timing,
        'missing_data' : missing_data,
        'dependency_check' : dependencies,
    }
    return str(res)#add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
