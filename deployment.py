'''
A deployment script that store best model and related files in deployment folder

Author: Mustafa Alturki
'''

import os
import json
import shutil


# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(), config['output_folder_path'])
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path']) 


def store_model_into_pickle(model):
    '''
    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    '''
    if not os.path.exists(prod_deployment_path):
        os.mkdir(prod_deployment_path)
    shutil.copy(os.path.join(output_model_path, "trainedmodel.pkl"),
        prod_deployment_path)
    shutil.copy(os.path.join(output_model_path, "latestscore.txt"),
        prod_deployment_path)
    shutil.copy(os.path.join(output_folder_path, "ingestedfiles.txt"),
        prod_deployment_path)

if __name__ == "__main__":
    store_model_into_pickle(None)
        
        

