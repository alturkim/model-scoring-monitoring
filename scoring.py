'''
A scoring script that run the fitted model on test data, caculate F1 score and save it

Author: Mustafa Alturki
'''
import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 

def score_model(model_path, data_path):
    '''
    This function takes a trained model, load test data,
    and calculate an F1 score for the model relative to the test data
    it writes the result to the latestscore.txt file
    '''
    df = pd.read_csv(data_path)
    model = pickle.load(open(model_path, "rb"))
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    x_data = df[features]
    y_data = df["exited"]

    predicted = model.predict(x_data)
    f1_score = metrics.f1_score(predicted, y_data)
    if not os.path.exists(os.path.join(output_model_path)):
        os.mkdir(os.path.join(output_model_path))
    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as out_f:
        out_f.write("F1 = " + str(f1_score) + "\n")
    return f1_score

if __name__ == "__main__":
    data_path = os.path.join(test_data_path, "testdata.csv")
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    score_model(model_path, data_path)