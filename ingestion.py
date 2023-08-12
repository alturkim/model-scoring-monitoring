import pandas as pd
import numpy as np
import os
from os import path
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    file_name_lst = [f for f in os.listdir(path.join(os.getcwd(), input_folder_path)) if f.endswith('.csv')]
    df_lst = []
    for file_name in file_name_lst:
        current_df = pd.read_csv(path.join(os.getcwd(), input_folder_path, file_name))
        df_lst.append(current_df)
    final_dataframe = pd.concat(df_lst, axis=0, ignore_index=True)
    final_dataframe = final_dataframe.drop_duplicates()
    if not path.exists(path.join(os.getcwd(), output_folder_path)):
        os.mkdir(path.join(os.getcwd(), output_folder_path))
    final_dataframe.to_csv(path.join(os.getcwd(), output_folder_path, "finaldata.csv"), index=False)
    
    with open(path.join(os.getcwd(), output_folder_path, "ingestedfiles.txt"), "w") as log_file:
        for file_name in file_name_lst:
            log_file.write(path.join(output_folder_path, file_name) + "\n")



if __name__ == '__main__':
    merge_multiple_dataframe()
