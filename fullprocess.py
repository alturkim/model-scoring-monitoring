import training
import scoring
import deployment
import diagnostics
import reporting
import os
import json
import subprocess
import pickle

with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path']) 
input_folder_path = os.path.join(os.getcwd(), config["input_folder_path"])
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 

# Check and read new data
# first, read ingestedfiles.txt
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as f:
    ingested_path_lst = [path.strip() for path in f if len(path) > 0]
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
data_path_lst = [f for f in os.listdir(os.path.join(os.getcwd(), input_folder_path)) if f.endswith('.csv')]
new_path_lst = [f for f in data_path_lst if f not in ingested_path_lst]


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if (len(new_path_lst) > 0):
    print("New data available, running the ingestion script")
    subprocess.call(['python', 'ingestion.py'])
else:
    print("No new data, exiting now ...")
    exit()

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
    latest_score = float(f.readline().split("=")[1].strip())
model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
new_score = scoring.score_model(model_path, os.path.join(output_folder_path, "finaldata.csv"))
drift = new_score < latest_score


# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if not drift:
    print("No model drift, exiting now ...")
    exit()


# Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
print("Model Drift Detected, Training new model ...")
subprocess.call(['python', 'training.py'])
print("Deploying the new model ...")
subprocess.call(['python', 'deployment.py'])

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
subprocess.call(['python', 'diagnostics.py'])
subprocess.call(['python', 'reporting.py'])
subprocess.call(['python', 'apicalls.py'])






