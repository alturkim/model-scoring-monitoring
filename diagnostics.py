'''
A diagnostics script that check execution time and data integrity and generate data summary.

Author: Mustafa Alturki
'''
import pandas as pd
import numpy as np
import timeit
import os
import subprocess
import json
import pickle

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path']) 

def model_predictions(df):
    '''read the deployed model and a test dataset, calculate predictions
    '''
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    x_data = df[features]
    y_data = df["exited"]
    model = pickle.load(
        open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb"))
    predicted = model.predict(x_data)

    return predicted.tolist()

def dataframe_summary():
    '''Function to get summary statistics
    '''
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    summary = []
    for col in features:
        col_stat = {"column": col}
        col_stat["mean"] = np.mean(df[col])
        col_stat["median"] = np.median(df[col])
        col_stat["std"] = np.std(df[col])
    summary.append(col_stat)
    return summary

def check_na():
    df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    summary = []
    for col in df.columns:
        na_count = df[col].isna().sum()
        na_perc = na_count / len(df[col])
        summary.append(na_perc*100)
    return summary

def execution_time():
    '''calculate timing of training.py and ingestion.py
    '''
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time =timeit.default_timer() - starttime
    
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time =timeit.default_timer() - starttime
    return [ingestion_time, training_time] 


def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    outdated = outdated.decode('utf-8').strip()
    package_lst, version_lst, latest_lst = [], [], []
    for line in outdated.split("\n")[2:]:
        package, version, latest, _ = line.split()
        package_lst.append(package)
        version_lst.append(version)
        latest_lst.append(latest)
    df = pd.DataFrame(list(zip(package_lst, version_lst, latest_lst)),
        columns=["Package", "Version", "Latest"])
    df.to_csv('outdated.txt', sep='\t', index=False)
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    model_predictions(df)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
