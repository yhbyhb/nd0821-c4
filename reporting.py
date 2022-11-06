import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def report_confusion_matrix():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    testdatacsv = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')
    testdata = pd.read_csv(testdatacsv)
    # X = trainingdata.loc[:,['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y_true = testdata['exited'].values
    y_pred = diagnostics.model_predictions(testdatacsv)
    # print(y_pred, y_true)
    
    cfm = metrics.confusion_matrix(y_true, y_pred)
    # print(cfm)

    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
    cfm_plot = sns.heatmap(df_cfm, annot=True)
    cfm_path = os.path.join(os.getcwd(), model_path, 'confusionmatrix.png')
    cfm_plot.figure.savefig(cfm_path)

if __name__ == '__main__':
    report_confusion_matrix()
