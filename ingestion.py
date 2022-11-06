import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    '''
    Function for data ingestion
    '''
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))

    df_list = pd.DataFrame()
    
    record_file_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')
    with open(record_file_path, 'w') as record_file:
        for each_filename in filenames:
            input_file_path = os.path.join(os.getcwd(), input_folder_path, each_filename)
            # print(input_file_path)
            data_frame = pd.read_csv(input_file_path)
            df_list=df_list.append(data_frame)

            record_file.write(each_filename + '\n')

    result=df_list.drop_duplicates()
    result.to_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
