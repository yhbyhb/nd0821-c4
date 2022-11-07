import json
import pickle
import os
import sys
import pandas as pd

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = config['output_folder_path']
ingested_file_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')
with open(ingested_file_path, 'r') as ingested_file:
    ingested = ingested_file.readlines()
    ingested = [x.strip() for x in ingested]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_dir = os.path.join(os.getcwd(), config['input_folder_path'])
input_files = os.listdir(input_dir)

all_input_files = list(set(ingested + input_files))
has_new_files = False;
if len(all_input_files) > len(ingested):
	has_new_files = True

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not has_new_files:
    print('there is no new file')
    sys.exit()

print('there is new files. Ingesting new files...')
new_df = ingestion.merge_multiple_dataframe()

prev_df = pd.read_csv(os.path.join(os.getcwd(), config['output_folder_path'], "finaldata.csv"))
new_df = prev_df.append(new_df, ignore_index=True)
new_df.drop_duplicates()
new_df.to_csv(os.path.join(os.getcwd(),config['output_folder_path'],"finaldata.csv"), index=False)

record_file_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')
with open(record_file_path, 'w') as record_file:
    for each_filename in all_input_files:
        record_file.write(each_filename + '\n')

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
latest_score_file = os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')
with open(latest_score_file, 'r') as score_file:
    prev_f1_score = float(score_file.readline())

model_file_path = os.path.join(os.getcwd(), prod_deployment_path)
new_f1_score = scoring.score_model(model_file_path, new_df)
model_drift = False
if new_f1_score < prev_f1_score:
    model_drift = True

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if not model_drift:
    print(f"No drift! prev f1 {prev_f1_score} < new f1 {new_f1_score}")
    sys.exit()
else:
    print(f"model drift! prev f1 {prev_f1_score} > new f1 {new_f1_score}")

print("Re-training...")
os.system('python3 training.py')


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
print('Re-deploying')
os.system('python3 deployment.py')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python3 diagnostics.py')
os.system('python3 reporting.py')
os.system('python3 apicalls.py')
