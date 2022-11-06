from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    model_path = os.path.join(config['output_model_path']) 
    output_folder_path = config['output_folder_path']

    deployment_path  = os.path.join(os.getcwd(), prod_deployment_path)

    model_file_path = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    latest_score_file = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    record_file_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')

    shutil.copy(model_file_path, deployment_path)
    shutil.copy(latest_score_file, deployment_path)
    shutil.copy(record_file_path, deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
