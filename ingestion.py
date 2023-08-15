'''
A simple ingestion script that combine multiple data sources, save them in a single file, and log info about the process

Author: Mustafa Alturki
'''

import pandas as pd
from datetime import datetime
import json
import os

# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(os.getcwd(), config['input_folder_path'])
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path'])

def merge_multiple_dataframe():
    """check for datasets, compile them together, and write to an output file
    """
    file_name_lst = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    df_lst = []
    for file_name in file_name_lst:
        current_df = pd.read_csv(os.path.join(input_folder_path, file_name))
        df_lst.append(current_df)
    final_dataframe = pd.concat(df_lst, axis=0, ignore_index=True)
    final_dataframe = final_dataframe.drop_duplicates()

    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    final_dataframe.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)
    
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as log_file:
        for file_name in file_name_lst:
            log_file.write(file_name + "\n")

if __name__ == '__main__':
    merge_multiple_dataframe()
