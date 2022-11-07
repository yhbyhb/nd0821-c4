import json
import os
import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"



#Call each API endpoint and store the responses
response_prediction = requests.post(URL+'/prediction?inputdata=testdata/testdata.csv').content
response_scoring = requests.get(URL+'/scoring').content
response_stats = requests.get(URL+'/summarystats').content
response_diag = requests.get(URL+'/diagnostics').content

#combine all API responses
responses = [response_prediction, response_scoring, response_stats, response_diag]

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 

apireturn_path = os.path.join(os.getcwd(), config['output_model_path'], 'apireturns.txt')
with open(apireturn_path, 'w') as file:
    file.write(str(responses))