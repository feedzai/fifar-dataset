import pandas as pd
import shutil
import subprocess
import yaml
import os
dataset_models_path = './ICAIF_KAGGLE'
expert_index = ['fpr_disparity.csv', 'full_w_table.csv']
case_index = ['deployment_predictions.csv','train_predictions.csv']
unindexed = ['p_of_error.csv','expert_properties.csv']

#Copy Datasets and Models to the main code structure

#Preprocessed Expert Data
if not os.path.isdir('./Code/experts/expert_info'):
    os.makedirs('./Code/experts/expert_info', exist_ok=True)
    for direc in os.listdir(dataset_models_path + '/experts/'):
        if direc.split('.')[-1] == 'csv':
            file = pd.read_csv(dataset_models_path + '/experts/'+direc)
            if direc in expert_index:
                file = file.rename(columns = {'Unnamed: 0':'expert'}).set_index('expert')
            if direc in case_index:
                file = file.set_index('case_id')
            if direc in unindexed:
                file = file.drop(columns = 'Unnamed: 0')
            print(f'done {direc}')
            file.to_parquet('./Code/experts/expert_info/'+ direc.split('.')[0] + '.parquet')
        else:
            shutil.copy(dataset_models_path + '/experts/'+direc, './Code/experts/expert_info/'+ direc)   
    #shutil.copytree(dataset_models_path + '/experts','./Code/experts/expert_info')

#ML Model
if not os.path.isdir('./Code/ml_model/model'):
    shutil.copytree(dataset_models_path + '/ml_model', './Code/ml_model/model')

#Expertise Model
if not os.path.isdir('./Code/expertise_models/models/small_regular/human_expertise_model'):
    shutil.copytree(dataset_models_path + '/human_expertise_model','./Code/expertise_models/models/small_regular/human_expertise_model')

#Dataset with limited expert predictions
if not os.path.isdir('./Code/testbed/train/small__regular'):
    os.makedirs('./Code/testbed/train/small__regular', exist_ok=True)
    for direc in os.listdir(dataset_models_path + '/testbed/train/small__regular'):
        if direc.split('.')[-1] == 'csv':
            file = pd.read_csv(dataset_models_path + '/testbed/train/small__regular/'+direc)
            if direc.split('.')[0] == 'batches':
                file.set_index('case_id')
            if direc.split('.')[0] == 'capacity':
                file.rename(columns = {'Unnamed: 0': 'batch_id'}).set_index('batch_id')
            if direc.split('.')[0] == 'train':
                file.set_index('case_id')
            file.to_parquet('./Code/testbed/train/small__regular/'+ direc.split('.')[0] + '.parquet')
        else:
            shutil.copy(dataset_models_path + '/testbed/train/small__regular/'+direc, './Code/testbed/train/small__regular/'+ direc)   
    #shutil.copytree(dataset_models_path + '/experts','./Code/experts/expert_info')

if not os.path.isdir('./Code/testbed/test'):
    os.makedirs('./Code/testbed/test', exist_ok=True)
    for direc in os.listdir(dataset_models_path + '/testbed/test'):
        if direc.split('.')[-1] == 'csv':
            file = pd.read_csv(dataset_models_path + '/testbed/test/'+direc)
            file = file.set_index('case_id')
            file.to_parquet('./Code/testbed/test/'+ direc.split('.')[0] + '.parquet')
        else:
            shutil.copytree(dataset_models_path + '/testbed/test/'+direc, './Code/testbed/test/'+ direc)   
    #shutil.copytree(dataset_models_path + '/experts','./Code/experts/expert_info')


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