from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
import diagnostics 
import scoring
# import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data_path = request.form['data_path']
    df = pd.read_csv(data_path)
    preds = diagnostics.model_predictions(df)
    return json.dumps(preds) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    data_path = os.path.join(test_data_path, "testdata.csv")
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    f1_score = scoring.score_model(model_path, data_path)
    return json.dumps(f1_score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    #check means, medians, and modes for each column
    stats_lst = diagnostics.dataframe_summary()
    return json.dumps(stats_lst) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    na_perc = diagnostics.check_na()
    ex_time = diagnostics.execution_time()
    dependancies_check = diagnostics.outdated_packages_list().to_dict()

    

    return {"NA Percentage": na_perc,
        "Ingestion Time": ex_time[0],
        "Training Time:": ex_time[1],
        "Dependacies Check": dependancies_check}


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
