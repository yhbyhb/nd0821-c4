import pickle
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

import ingestion
import training
import utils

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
test_data_csv = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')

##################Function to get model predictions
def model_predictions(test_file_path):
    #read the deployed model and a test dataset, calculate predictions
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    model = utils.load_model(prod_deployment_path)

    testdata = pd.read_csv(test_file_path)
    X, _ = utils.split_data(testdata)

    predicted = model.predict(X)

    return predicted

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data_frame = pd.read_csv(data_path)

    data_frame, _  = utils.split_data(data_frame)

    result_summary = data_frame.agg(["mean", "median", "std"])

    statistics = list(result_summary['lastmonth_activity']) + \
                 list(result_summary['lastyear_activity']) + \
                 list(result_summary['number_of_employees'])

    # print(statistics)
    return statistics#return value should be a list containing all summary statistics

##################Function to get percent of missing data
def missing_data():
    data_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data_frame = pd.read_csv(data_path)

    missing_values_df = data_frame.isna().sum() / data_frame.shape[0]
    return missing_values_df.values.tolist()

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    iter = 1000
    starttime = timeit.default_timer()
    for i in range(iter):
        ingestion.merge_multiple_dataframe()
    ingestion_timing = (timeit.default_timer() - starttime) / iter

    starttime = timeit.default_timer()
    for i in range(iter):
        training.train_model(dataset_csv_path, model_path)
    training_timing = (timeit.default_timer() - starttime) / iter
    # print(ingestion_timing, training_timing)
    return ingestion_timing, training_timing  #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    data_frame = pd.DataFrame(columns=['package_name', 'current', 'recent_available'])

    with open("requirements.txt", "r") as file:
        strings = file.readlines()
        # print(strings)
        names = []
        cur = []
        recent = []
        # strings.sort()

        for line in strings:
            name, cur_ver = line.strip().split('==')
            # print(words[0], words[1])
            names.append(name)
            cur.append(cur_ver)
            # info = subprocess.check_output(['pip', 'show', name)
            info = subprocess.check_output(['python', '-m', 'pip', 'show', name])
            recent.append(str(info).split('\\n')[1].split()[1])
        
        data_frame['package_name'] = names
        data_frame['current'] = cur
        data_frame['recent_available'] = recent

    # print(data_frame.values.tolist())
    return data_frame.values.tolist()


if __name__ == '__main__':
    model_predictions(test_data_csv)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
    
