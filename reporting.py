import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 

def score_model():
    '''
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    '''
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y_data = df["exited"]
    predicted = model_predictions(df)
    cm = metrics.confusion_matrix(y_data, predicted)
    plot = metrics.ConfusionMatrixDisplay(cm).plot()
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))

if __name__ == '__main__':
    score_model()
