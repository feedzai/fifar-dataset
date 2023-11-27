import json
import pickle

import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import recall_score
import lightgbm as lgb
import os



def tpr_at_fpr(labels, preds):
    is_higher_better = True
    results = pd.DataFrame()
    results["true"] = labels
    results["score"] = preds

    temp = results.sort_values(by="score", ascending=False)

    FPR = 0.05
    N = (temp["true"] == 0).sum()
    FP = round(FPR * N)
    aux = temp[temp["true"] == 0]

    threshold = aux.iloc[FP - 1, 1]

    y_pred = np.where(results["score"] >= threshold, 1, 0)
    tpr = recall_score(labels, y_pred)
    return "tpr_at_fpr", tpr, is_higher_better

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends the default JSONEncoder to handle NumPy data types.

    This encoder converts NumPy integers and floats to Python built-in types, and NumPy
    arrays to Python lists, so they can be encoded as JSON strings. All other objects
    are handled by the default JSONEncoder.

    Attributes:
        None

    Methods:
        default(obj): Overrides the default method to handle NumPy data types.

    Usage:
        Use this encoder in conjunction with the json.dumps() function to encode objects
        that contain NumPy data types as JSON strings.

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class HPO:

    def __init__(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        method="TPE",
        parameters=None,
        path="",):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.method = method
        self.parameters = parameters
        self.path = path
        self.current_best = -np.inf
        print(f"{self.path}/best_model.pickle", "wb")

    def initialize_optimizer(self, categorical, n_jobs):
        self.catcols = categorical
        self.tpe_optimization(n_jobs)
    
    def tpe_optimization(self, n_jobs):

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed = 42)
        )

        study.set_user_attr("X_train", self.X_train)
        study.set_user_attr("X_val", self.X_val)
        study.set_user_attr("y_train", self.y_train)
        study.set_user_attr("y_val", self.y_val)
        study.set_user_attr("n_jobs", n_jobs)

        optuna.logging.set_verbosity(optuna.logging.FATAL)
        study.optimize(self.objective, n_trials=100)

        trials_df = study.trials_dataframe()
        keys = trials_df.keys()

        param_list = []
        for i in range(len(trials_df)):
            aux_dict = {}
            for key in keys:
                if key in ["datetime_start", "datetime_complete", "duration"]:
                    continue
                aux_dict[key] = trials_df[key].iloc[i]
            param_list.append(aux_dict)

        with open(
            "{}/config.json".format(self.path), "w"
        ) as fout:
            # Write the dictionary to the file
            json.dump(param_list, fout, cls=NpEncoder, indent=4)
        return


    def objective(self, trial):  # Instance method
        """Optuna objective function for LightGBM model hyperparameter optimization.

        Args:
            trial: An Optuna trial object.

        Returns:
            The true positive rate (recall score) on the validation set, to maximize.
        """

        X_train = trial.study.user_attrs["X_train"]
        X_val = trial.study.user_attrs["X_val"]
        y_train = trial.study.user_attrs["y_train"]
        y_val = trial.study.user_attrs["y_val"]
        n_jobs = trial.study.user_attrs["n_jobs"]
        
        # Define the hyperparameters to optimize
        max_depth = trial.suggest_int("max_depth", 2, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 5000, log=True)
        num_leaves = trial.suggest_int("num_leaves", 10, 1000, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 500, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0001, 0.1, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0001, 0.1, log=True)

        # Train the model with the given hyperparameters
        model = lgb.LGBMClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            num_leaves=num_leaves,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            boosting_type="goss",
            enable_bundle=False,
            seed = 42
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val),
            ],
            eval_metric=tpr_at_fpr,
            early_stopping_rounds=10,
            categorical_feature= 'auto',
            verbose=False
        )

        # Evaluate the model on the testing data
        y_pred = model.predict_proba(X_val)
        y_pred = y_pred[:, 1]

        # GET THRESHOLD at 5% fpr
        results = pd.DataFrame()
        results["true"] = y_val
        results["score"] = y_pred

        temp = results.sort_values(by="score", ascending=False)

        FPR = 0.05
        N = (temp["true"] == 0).sum()
        FP = round(FPR * N)
        aux = temp[temp["true"] == 0]

        threshold = aux.iloc[FP - 1, 1]

        y_pred = np.where(results["score"] >= threshold, 1, 0)
        tpr = recall_score(y_val, y_pred)

        if tpr > self.current_best:
            self.current_best = tpr
            os.makedirs(self.path, exist_ok = True)
            with open(f"{self.path}/best_model.pickle", "wb") as fout:
                pickle.dump(model, fout)

        return tpr  
    