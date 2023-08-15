'''
A training script that fits a logistic regression model and save the trained model

Author: Mustafa Alturki
'''
import pickle
import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path'])
model_path = os.path.join(os.getcwd(), config['output_model_path'])

def train_model():
    '''
    Function for training the model
    '''
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    x_data = df[features]
    y_data = df["exited"]
    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    model.fit(x_data, y_data)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    pickle.dump(model, open(os.path.join(model_path, "trainedmodel.pkl"), "wb"))

if __name__ == "__main__":
    train_model()
