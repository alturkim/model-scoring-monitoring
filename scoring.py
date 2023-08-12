from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data,
    #and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    x_data = df[features]
    y_data = df["exited"]

    model = pickle.load(open(os.path.join(output_model_path, "trainedmodel.pkl"), "rb"))
    
    predicted = model.predict(x_data)
    f1_score = metrics.f1_score(predicted, y_data)
    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as out_f:
        out_f.write("F1 = " + str(f1_score) + "\n")

if __name__ == "__main__":
    score_model()