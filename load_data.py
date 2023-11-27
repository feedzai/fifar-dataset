import pandas as pd
import shutil
import subprocess
import yaml
import os
dataset_models_path = ''

#Copy Datasets and Models to the main code structure

#Preprocessed Expert Data
if not os.path.isdir('./Code/experts/expert_info'):   
    shutil.copytree(dataset_models_path + '/experts/expert_info','./Code/experts/expert_info')

#ML Model
if not os.path.isdir('./Code/ml_model/model'):  
    shutil.copytree(dataset_models_path + '/ml_model/model', './Code/ml_model/model')

#Expertise Model
if not os.path.isdir('./Code/expertise_models/models/small_regular/human_expertise_model'):
    shutil.copytree(dataset_models_path + '/expertise_models/models/small_regular/human_expertise_model','./Code/expertise_models/models/small_regular/human_expertise_model')

#Dataset with limited expert predictions
if not os.path.isdir('./Code/testbed/train/small#regular'):
    shutil.copytree(dataset_models_path + '/testbed/train/small#regular', './Code/testbed/train/small#regular')

if not os.path.isdir('./Code/testbed/test'):
    shutil.copytree(dataset_models_path + '/testbed/test', './Code/testbed/test')

Input_Data = pd.read_csv('./Code/data/Base.csv')

Input_Data.sort_values(by = 'month', inplace = True)
Input_Data.reset_index(inplace=True)
Input_Data.drop(columns = 'index', inplace = True)
Input_Data.index.rename('case_id', inplace=True)

data_cfg_path = './Code/data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

Input_Data.loc[:,data_cfg['data_cols']['categorical']] = Input_Data.loc[:,data_cfg['data_cols']['categorical']].astype('category')

Input_Data.to_parquet('./Code/data/BAF.parquet')

subprocess.run(["python", "./Code/ml_model/training_and_predicting.py"])






