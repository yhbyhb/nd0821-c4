import os
import pickle

def load_model(model_path):
    model_file_path = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def split_data(data_frame):
    feature_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    dropped = data_frame.drop('corporation', axis=1)
    return dropped[feature_cols], dropped['exited']