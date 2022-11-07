import json
import glob
import os
import pandas as pd


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))

    datasets = glob.glob(f'{input_folder_path}/*.csv')
    df = pd.concat(map(pd.read_csv, datasets))

    record_file_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')
    with open(record_file_path, 'w') as record_file:
        for each_filename in filenames:
            record_file.write(each_filename + '\n')

    result = df.drop_duplicates()
    result.to_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'), index=False)

    return result

if __name__ == '__main__':
    merge_multiple_dataframe()
