# %%
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
import json

sns.set_style('whitegrid')

cfg_path = './cfg.yaml'

with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

with open(cfg['data_cfg_path'], 'r') as infile:
    data_cfg = yaml.safe_load(infile)

cfg = cfg['test_run']

RESULTS_PATH = cfg['results_path']
MODELS_PATH = cfg['model_path']


cat_dict = data_cfg['categorical_dict']


with open(cfg['metadata'], 'r') as infile:
    metadata = yaml.safe_load(infile)

LABEL_COL = 'fraud_label'
PROTECTED_COL = metadata['data_cols']['protected']
CATEGORICAL_COLS = metadata['data_cols']['categorical']
TIMESTAMP_COL = metadata['data_cols']['timestamp']
SCORE_COL = metadata['data_cols']['score']
BATCH_COL = metadata['data_cols']['batch']
ASSIGNMENT_COL = metadata['data_cols']['assignment']
DECISION_COL = metadata['data_cols']['decision']
EXPERT_IDS = metadata['expert_ids']

print(EXPERT_IDS)

# %%
RMAs = dict()

for train_env in os.listdir(MODELS_PATH):
    RMAs[train_env] = haic.assigners.RiskMinimizingAssigner(
        expert_ids=EXPERT_IDS,
        outputs_dir=f'{MODELS_PATH}{train_env}/human_expertise_model/',
    )
    calibrator_path = f'{MODELS_PATH}{train_env}/human_expertise_model/calibrator.pickle'
    RMAs[train_env].load(CATEGORICAL_COLS, SCORE_COL, ASSIGNMENT_COL, calibrator_path, cat_dict)

#FIELDS describes all the experiment parameters
ENV_FIELDS = ['batch', 'capacity']
ASSIGNER_FIELDS = [
    'confidence_deferral', 'solver', 'calibration', 'fp_cost', 'fp_protected_penalty',
    'dynamic', 'target_fpr_disparity', 'fpr_learning_rate', 'fpr_disparity_learning_rate'
]
FIELDS = ENV_FIELDS + ASSIGNER_FIELDS

file = open("../ml_model/model/model_properties.pickle", "rb")
model_properties = pickle.load(file)
file.close()

ML_MODEL_THRESHOLD = model_properties['threshold']
THEORETICAL_FP_COST = ML_MODEL_THRESHOLD/(1-ML_MODEL_THRESHOLD)


test = pd.read_parquet(cfg['test_paths']['data'])

test_experts_pred = pd.read_parquet(cfg['test_paths']['experts_pred'])

TEST_X = test.drop(columns=[TIMESTAMP_COL, LABEL_COL])

test_experts_pred_thresholded = test_experts_pred.copy()

test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] = (
        test_experts_pred_thresholded[EXPERT_IDS['model_ids'][0]] >= ML_MODEL_THRESHOLD
).astype(int)


test_eval = haic.HAICEvaluator(
    y_true=test[LABEL_COL],
    experts_pred=test_experts_pred,
    exp_id_cols=FIELDS
)

def make_id_str(tpl):
    printables = list()
    for i in tpl:
        if i == '':
            continue
        elif isinstance(i, (bool, int, float)):
            printables.append(str(i))
        else:
            printables.append(i)

    return '_'.join(printables)


def make_assignments(X, envs, rma, exp_params, test_env_id):
    assigner_params = {k: v for k, v in exp_params.items() if k not in ['batch', 'capacity']}
    params_to_record = {k: exp_params[k] for k in FIELDS}
    exp_id = tuple([v for k, v in params_to_record.items()])
    rel_path = make_id_str(exp_id) + '_' + test_env_id[0] + '_' + test_env_id[1] + '.pkl'
    a = rma.assign(
        X=X, score_col=SCORE_COL,
        batches=envs[test_env_id]['batches'],
        capacity=envs[test_env_id]['capacity'].T.to_dict(),
        ml_model_threshold=ML_MODEL_THRESHOLD,
        protected_col=(X[PROTECTED_COL] >= 50).map({True: 'Older', False: 'Younger'}),
        protected_group='Older',
        assignments_relative_path=rel_path,
        **assigner_params
    )

    return exp_id, assigner_params, a

def get_batches(batch_df, assignments, decisions, test_experts_pred, exp_id_cols, exp_id, THEORETICAL_FP_COST):
    nbatches = batch_df['batch_id'].max()
    batch_loss = np.zeros(nbatches)
    n_assignments = len(batch_df)
    
    for i in range(1, nbatches+1):
        index_list = batch_df.index[batch_df['batch_id'] == i].to_list()
        
        batch_eval = haic.HAICEvaluator(
            y_true=test[LABEL_COL].loc[index_list],
            experts_pred=test_experts_pred.loc[index_list],
            exp_id_cols=exp_id_cols
        )

        batch_eval.evaluate(
            exp_id=exp_id,
            assignments=assignments.loc[index_list],
            decisions=decisions.loc[index_list],
            assert_capacity_constraints=False
        )
        
        batch_results = batch_eval.get_results(short = False)
        batch_results['loss'] = (THEORETICAL_FP_COST * batch_results['fp'] + batch_results['fn']).astype('float')
        batch_loss[i-1] = batch_results['loss']

    batch_stats = {'AVG_Batch_Loss':np.mean(batch_loss),
                   'STD_Batch_Loss':np.std(batch_loss),
                   'AVG_Batch_Loss/Batch_size': np.mean(batch_loss)/(n_assignments/nbatches),
                   'N_batches':nbatches
                   }
    
    return batch_stats


# %%
to_test = list()

BASE_CFG = cfg['base_cfg']

def product_dict(**kwargs):  # aux
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def make_params_combos(params_cfg):
    params_list = list()
    if not isinstance(params_cfg, list):
        params_cfg = [params_cfg]

    for cartesian_product_set in params_cfg:
        for k, v in cartesian_product_set.items():
            if isinstance(v, str):
                cartesian_product_set[k] = [v]
        for p in product_dict(**cartesian_product_set):
            p_params = {**BASE_CFG, **p}
            if p_params['fp_cost'] == 'theoretical':
                p_params['fp_cost'] = THEORETICAL_FP_COST
            if p_params['confidence_deferral']:
                params_list.append(p_params)
            if (not p_params['confidence_deferral']) and p_params['solver'] == 'scheduler':
                params_list.append(p_params)
            if (not p_params['confidence_deferral']) and p_params['solver'] == 'individual':
                params_list.append(p_params)

    return params_list

to_test = make_params_combos(cfg['experiments'])

TEST_ENVS = dict()

batches_cap_path = './test/'

#Just need to make sure tht whatever ends up in TEST_ENVS is the same thing. So basically they can be saved and viewed in an appealing fashion but they
#Must actually go into the model in the proper format which is one column per expert (model+human)
for dir in os.listdir(batches_cap_path):
    if(os.path.isfile(batches_cap_path + dir)):
        continue


    batch_name = dir.split('#')[0]
    cap_name = dir.split('#')[1]

    bat = pd.read_csv(batches_cap_path + dir + '/batches.csv')
    bat = bat.set_index('case_id')

    cap = pd.read_csv(batches_cap_path + dir + '/capacity.csv')
    cap = cap.set_index('batch_id')

    model_cap = cap['batch_size'] - cap.drop(columns = 'batch_size').sum(axis = 1)

    cap['batch_size'] = model_cap
    cap = cap.rename(columns = {'batch_size': 'model#0'})
    TEST_ENVS[tuple([batch_name, cap_name])] = { 
        'batches': bat,
        'capacity': cap,
        }

for env_id, rma in RMAs.items():
    rma.outputs_dir = './test_results/' + env_id + '/'
    os.makedirs(rma.outputs_dir, exist_ok = True)


test_env_df = pd.DataFrame(columns = ['batch_size','batch_seed', 'absence_rate', 'absence_seed', 'distribution', 'distribution_std', 'distribution_seed', 'deferral_rate', 'exp_pool','fpr_disp'])


# %%

batch_results_df = pd.DataFrame()
TEST_ENVS_L = dict()
if cfg['n_jobs'] > 1:
    for exp_params in to_test:
        for test_env_id in TEST_ENVS:
            assigner_params = {k: v for k, v in exp_params.items() if k not in ['batch', 'capacity']}
            params_to_record = {k: exp_params[k] for k in FIELDS}
            exp_id = tuple([v for k, v in params_to_record.items()])
            rel_path = make_id_str(exp_id) + '_' + test_env_id[0] + '_' + test_env_id[1] + '.pkl'
            if rel_path not in os.listdir('./test_results/small_regular/'):
                TEST_ENVS_L[test_env_id] = TEST_ENVS[test_env_id]
        Parallel(n_jobs=cfg['n_jobs'])(
            delayed(make_assignments)(
                X=TEST_X,
                envs=TEST_ENVS,
                rma=RMAs[exp_params['batch']+ '_' + exp_params['capacity']],
                exp_params=exp_params,
                test_env_id = test_env_id
            )
            for test_env_id in TEST_ENVS_L
        )
        
else:
    for exp_params in to_test:
        for test_env_id in TEST_ENVS:
            print(test_env_id)
            if test_env_id[0].split('_')[0] == 'large':
                batch_size = 5000
            elif test_env_id[0].split('_')[0] == 'small':
                batch_size = 250

            batch_seed = test_env_id[0].split('-')[1]

            if test_env_id[1].split('_')[0] == 'homogenous':
                distribution = 'homogenous'
                distribution_seed = 'NA'
                distribution_std = 'NA'
            else:
                distribution = 'variable'
                distribution_seed = test_env_id[1].split('_')[0].split('-')[1]
                distribution_std = '0.2'

            if test_env_id[1].split('_')[1] == 'fullteam':
                absence = 0
                absence_seed = 'NA'
            else:
                absence = 0.5
                absence_seed = test_env_id[1].split('_')[1].split('-')[1]
            
            if test_env_id[1].split('_')[2] == 'def20':
                deferral_rate = 0.2
            else:
                deferral_rate = 0.5

            if test_env_id[1].split('_')[-1] == 'sp':
                exp_pool = 'sparse'
            elif test_env_id[1].split('_')[-1] == 'ma':
                exp_pool = 'agreeing'
            elif test_env_id[1].split('_')[-1] == 'un':
                exp_pool = 'unfair'
            elif test_env_id[1].split('_')[-1] == 'st':
                exp_pool = 'standard'
            else:
                exp_pool = 'all'
            
            exp_id, assigner_params, a = make_assignments(
                X=TEST_X,
                envs=TEST_ENVS,
                rma=RMAs[exp_params['batch']+ '_' + exp_params['capacity']],
                exp_params=exp_params,
                test_env_id = test_env_id
            )
            print(exp_id)

            d = haic.query_experts(
                pred=test_experts_pred_thresholded,
                assignments=a
            )
            test_eval.evaluate(
                exp_id=exp_id,
                assignments=a,
                decisions=d,
                assert_capacity_constraints=False
            )
            print(exp_pool)

            old_ix = TEST_X.loc[TEST_X['customer_age'] >= 50].index
            yng_ix = TEST_X.loc[TEST_X['customer_age'] < 50].index

            label = test[LABEL_COL]

            old_pred = d.loc[old_ix]
            old_label = label.loc[old_ix]
            fp_old = ((old_pred == 1) & (old_label == 0)).astype(int).sum()
            tn_old = ((old_pred == 0) & (old_label == 0)).astype(int).sum()

            yng_pred = d.loc[yng_ix]
            yng_label = label.loc[yng_ix]
            fp_yng = ((yng_pred == 1) & (yng_label == 0)).astype(int).sum()
            tn_yng = ((yng_pred == 0) & (yng_label == 0)).astype(int).sum()

            fpr_yng = fp_yng/(fp_yng + tn_yng)
            fpr_old = fp_old/(fp_old + tn_old)

            fpr_disp =  fpr_yng/fpr_old
            test_env_df = test_env_df.append(pd.Series([batch_size, batch_seed, absence, absence_seed, distribution, distribution_std, distribution_seed, deferral_rate, exp_pool, fpr_disp], index = test_env_df.columns), ignore_index = True)


print(test_env_df)
test_results = test_eval.get_results(short=False)
test_results['loss'] = (THEORETICAL_FP_COST * test_results['fp'] + test_results['fn']).astype('float')
test_results = pd.concat([test_results, test_env_df], axis = 1, join = 'inner')

test_results.to_parquet('./test_results/test_results_08_15_2023.parquet')

