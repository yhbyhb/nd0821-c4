import pickle
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
testdatacsv = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')

##################Function to get model predictions
def model_predictions(testdatacsv):
    #read the deployed model and a test dataset, calculate predictions
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 

    model_file_path = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

    testdata = pd.read_csv(testdatacsv)
    X = testdata[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)

    predicted = model.predict(X)
    return predicted

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data_frame = pd.read_csv(data_path)

    statistics = [
        np.mean(data_frame['lastmonth_activity']),
        np.median(data_frame['lastmonth_activity']),
        np.std(data_frame['lastmonth_activity']),
        np.mean(data_frame['lastyear_activity']),
        np.median(data_frame['lastyear_activity']),
        np.std(data_frame['lastyear_activity']),
        np.mean(data_frame['number_of_employees']),
        np.median(data_frame['number_of_employees']),
        np.std(data_frame['number_of_employees']),
        ]
    # print(statistics)
    return statistics#return value should be a list containing all summary statistics

##################Function to get percent of missing data
def missing_data():
    data_path = os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv')
    data_frame = pd.read_csv(data_path)

    nas=list(data_frame.isna().sum())
    napercents=[nas[i]/len(data_frame.index) for i in range(len(nas))]
    # print(napercents)
    return napercents

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - starttime
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
    model_predictions(testdatacsv)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
    
