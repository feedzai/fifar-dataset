import os
import itertools
#hi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from sklearn import metrics
from aequitas.group import Group

from autodefer.models import haic
from autodefer.utils import thresholding as t, plotting

import pickle

sns.set_style('whitegrid')


cfg_path ='cfg.yaml'

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

RESULTS_PATH = cfg['results_path'] + '/'
MODELS_PATH = cfg['models_path']  + '/'

data_cfg_path = '../data/dataset_cfg.yaml'
with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']


os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

with open(cfg['metadata'], 'r') as infile:
    metadata = yaml.safe_load(infile)

LABEL_COL = metadata['data_cols']['label']
PROTECTED_COL = metadata['data_cols']['protected']
CATEGORICAL_COLS = metadata['data_cols']['categorical']
TIMESTAMP_COL = metadata['data_cols']['timestamp']

SCORE_COL = metadata['data_cols']['score']
BATCH_COL = metadata['data_cols']['batch']
ASSIGNMENT_COL = metadata['data_cols']['assignment']
DECISION_COL = metadata['data_cols']['decision']

EXPERT_IDS = metadata['expert_ids']

TRAIN_ENVS = {
    tuple(exp_dir.split('#')): {
        'train': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/train.parquet'),
        'batches': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/batches.parquet'),
        'capacity': pd.read_parquet(cfg['train_paths']['environments'] + exp_dir + '/capacity.parquet'),
    }
    for exp_dir in os.listdir(cfg['train_paths']['environments'])
    if os.path.isdir(cfg['train_paths']['environments']+exp_dir)
}

# DEFINING Lambda (fp cost) ---------------------------------------------------------------------------------

with open(f'../ml_model/model/model_properties.pickle', 'rb') as infile:
    ml_model_properties = pickle.load(infile)

ml_model_threshold = ml_model_properties['threshold']
ml_model_recall = 1 - ml_model_properties['fnr']
ml_model_fpr_diff = ml_model_properties['disparity']
ml_model_fpr = ml_model_properties['fpr']

#Our defined lambda
THEORETICAL_FP_COST = -ml_model_threshold / (ml_model_threshold - 1)

# Training our Expertise Model. A user can train this model under various training conditions, defined in testbed_train_generation.py
VAL_ENVS = dict()
VAL_X = None
RMAs = dict()
for env_id in TRAIN_ENVS:
    batch_id, capacity_id = env_id
    models_dir = f'{MODELS_PATH}{batch_id}_{capacity_id}/'
    os.makedirs(models_dir, exist_ok=True)

    train_with_val = TRAIN_ENVS[env_id]['train']
    train_with_val = train_with_val.copy().drop(columns=BATCH_COL)

    #Possibly subsample here


    is_val = (train_with_val[TIMESTAMP_COL] == 6)
    train_with_val = train_with_val.drop(columns=TIMESTAMP_COL)
    train = train_with_val[~is_val].copy()
    val = train_with_val[is_val].copy()

    RMAs[env_id] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{models_dir}human_expertise_model/',
    )

    RMAs[env_id].fit(
        train=train,
        val=val,
        categorical_cols=CATEGORICAL_COLS, score_col=SCORE_COL,
        decision_col=DECISION_COL, ground_truth_col=LABEL_COL, assignment_col=ASSIGNMENT_COL,
        hyperparam_space=cfg['human_expertise_model']['hyperparam_space'],
        n_trials=cfg['human_expertise_model']['n_trials'],
        random_seed=cfg['human_expertise_model']['random_seed'], 
        CAT_DICT = cat_dict
    )