import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


#Call each API endpoint and store the responses
response1 = requests.post(URL+'/prediction', data={"data_path": "testdata/testdata.csv"}).json()
response2 = requests.get(URL+'/scoring').json()
response3 = requests.get(URL+'/summarystats').json()
response4 = requests.get(URL+'/diagnostics').json()


#combine all API responses
responses = {'Prediction Result': response1,
    'Scoring Result': response2,
    'Summary Stats': response3,
    'Diagnostics Results': response4}

#write the responses to your workspace

with open('config.json','r') as f:
    config = json.load(f) 
with open(os.path.join(os.getcwd(), config["output_model_path"], "apireturns.txt"), "w") as f:
    json.dump(responses, f)


