# %%
import pandas as pd
import yaml
import numpy as np
import hpo_fpr
import os
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

data_cfg_path = (Path(__file__).parent/'../data/dataset_cfg.yaml').resolve()
cfg_path = Path(__file__).parent/'cfg.yaml'

with open(data_cfg_path, 'r') as infile:
    data_cfg = yaml.safe_load(infile)

#cfg_path = 'cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

cat_dict = data_cfg['categorical_dict']

def cat_checker(data, features, cat_dict):
    new_data = data.copy()
    for feature in features:
        if new_data[feature].dtype.categories.to_list() != cat_dict[feature]:
            new_data[feature] = pd.Categorical(new_data[feature].values, categories=cat_dict[feature])
    
    return new_data

# DATA LOADING -------------------------------------------------------------------------------------
data = pd.read_parquet(Path(__file__).parent/'../data/BAF.parquet')
LABEL_COL = data_cfg['data_cols']['label']
TIMESTAMP_COL = data_cfg['data_cols']['timestamp']
PROTECTED_COL = data_cfg['data_cols']['protected']
CATEGORICAL_COLS = data_cfg['data_cols']['categorical']

data = cat_checker(data, CATEGORICAL_COLS, cat_dict)

def splitter(df, timestamp_col, beginning: int, end: int):
    return df[
        (df[timestamp_col] >= beginning) &
        (df[timestamp_col] < end)].copy()

train = splitter(data, TIMESTAMP_COL, *cfg['splits']['train']).drop(columns=TIMESTAMP_COL)
ml_val = splitter(data, TIMESTAMP_COL, *cfg['splits']['ml_val']).drop(columns=TIMESTAMP_COL)
deployment = splitter(data, TIMESTAMP_COL, *cfg['splits']['deployment']).drop(columns=TIMESTAMP_COL)


# %%
def fpr_thresh(y_true, y_pred):
    results = pd.DataFrame()
    results["true"] = y_true
    results["score"] = y_pred

    temp = results.sort_values(by="score", ascending=False)

    FPR = 0.05
    N = (temp["true"] == 0).sum()
    FP = round(FPR * N)
    aux = temp[temp["true"] == 0]

    threshold = aux.iloc[FP - 1, 1]

    y_pred = np.where(results["score"] >= threshold, 1, 0)
    tpr = metrics.recall_score(y_true, y_pred)

    return tpr, threshold

X_train = train.drop(columns = 'fraud_bool')
y_train = train['fraud_bool']

X_val = ml_val.drop(columns = 'fraud_bool') 
y_val = ml_val['fraud_bool']

if not (os.path.exists(Path(__file__).parent/'./model/best_model.pickle')):
    opt = hpo_fpr.HPO(X_train,X_val,y_train,y_val, method = 'TPE', path = f"./model")
    opt.initialize_optimizer(CATEGORICAL_COLS, 25)

with open(Path(__file__).parent/'./model/best_model.pickle', 'rb') as infile:
        model = pickle.load(infile)

y_pred = model.predict_proba(X_val)
y_pred = y_pred[:,1]
roc_curve_clf = dict()
rec_at_5, thresh = fpr_thresh(y_val, y_pred)


X_test = deployment.drop(columns = 'fraud_bool')
y_test = deployment['fraud_bool']
y_pred = model.predict_proba(X_test)
y_pred = y_pred[:,1]

roc_curve_clf = dict()
roc_curve_clf['fpr'],roc_curve_clf['tpr'],roc_curve_clf['thr'] = metrics.roc_curve(y_test, y_pred)
pred = np.where(y_pred >= thresh, 1, 0)
tpr = metrics.recall_score(y_test, pred)

deployment['model_score'] = y_pred
deployment.to_parquet(Path(__file__).parent/'../data/BAF_deployment_score.parquet')

model_perf_test = pd.DataFrame(index = deployment.index)
model_perf_test['model_pred'] = (deployment['model_score'] >= thresh).astype(int)
model_perf_test['label'] = deployment['fraud_bool']

tn, fp, fn, tp = confusion_matrix(model_perf_test['label'], model_perf_test['model_pred']).ravel()
fpr_dep = fp/(fp + tn)
model_perf_test_o = model_perf_test.loc[deployment['customer_age']>= 50]
model_perf_test_y = model_perf_test.loc[deployment['customer_age']< 50]



tn, fp, fn, tp = confusion_matrix(model_perf_test_o['label'], model_perf_test_o['model_pred']).ravel()
fpr_o = fp/(fp+tn)
tn, fp, fn, tp = confusion_matrix(model_perf_test_y['label'], model_perf_test_y['model_pred']).ravel()
fpr_y = fp/(fp+tn)

disparity = (fpr_o-fpr_y)

model_properties = {'fpr':0.05,
                    'fnr': 1 - tpr,
                    'threshold': thresh,
                    'disparity': disparity}


file_to_store = open(Path(__file__).parent/"./model/model_properties.pickle", "wb")
pickle.dump(model_properties, file_to_store)
file_to_store.close()


