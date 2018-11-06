import time
import pandas as pd
import numpy as np
import xgboost as xgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from lib.util import *
from typing import Tuple, List, Dict

@timeit
def train_svm(X: pd.DataFrame, y: pd.Series, config: Config, params: Dict):
    # cv
    nfold = 5
    if config["mode"] == 'classification':
        skf = StratifiedKFold(n_splits = nfold, shuffle=True, random_state=777)
    else:
        skf = KFold(n_splits = nfold, shuffle=True, random_state=777)
    skf_split = skf.split(X, y)
    
    cv_res = []
    stime = time.time()
    for fid, (train_idx, valid_idx) in enumerate(skf_split):
        print("FoldID:{}".format(fid))
        fb_time = time.time()
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        if config.is_classification():
            clf = SVC() 
        else:
            clf = SVR()
        clf.set_params(**params)
        cur_model = clf.fit(X_train, y_train)
        config["model"]["svm"].append(cur_model)
        
        pred = cur_model.predict(X_valid)
        if config.is_classification():
            cv_res.append(roc_auc_score(y_valid, pred))
        else:
            cv_res.append(np.sqrt(((pred - y_valid) ** 2).mean()))

        fe_time = time.time() 
        if fe_time - fb_time >= 240: # 4min
            break
        if (fe_time - stime) / (fid + 1) * 2.0 >= config.time_left():
            break
    return np.mean(cv_res)


@timeit
def predict_svm(X: pd.DataFrame, config: Config) -> List:
    ans = []
    for model in config["model"]["svm"]:
        p = model.predict(X)
        if config["non_negative_target"]:
            p = [max(0, i) for i in p]
        ans.append(p)
    return np.mean(ans, axis = 0).flatten()


@timeit
def hyperopt_svm(X: pd.DataFrame, y: pd.Series, config: Config, max_trials: int=50):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.25)

    params = {
        "verbose": 0,
    }

    clf_space = {
        "C": hp.uniform("C", 1.0, 10.0),
        "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(1e-2)),
        "kernel": hp.choice("kernel", ["rbf", "poly", "sigmoid", "linear"]),
        "degree": hp.choice("degree", range(2, 8)),
        "tol": hp.loguniform("tol", np.log(1e-4), np.log(1.0)),
        "shrinking": hp.choice("shrinking", [True, False]),
        "max_iter": hp.choice("max_iter", range(500, 2500, 100)),
    }
    
    reg_space = {
        "kernel": hp.choice("kernel", ["rbf", "poly", "sigmoid", "linear"]),
        "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(1e-2)),
        "degree": hp.choice("degree", range(2, 8)),
        "tol": hp.loguniform("tol", np.log(1e-4), np.log(1.0)),
        "epsilon": hp.uniform("epsilon", 0.0, 0.2),
        "C": hp.loguniform("C", np.log(0.5), np.log(10.0)),
        "shrinking": hp.choice("shrinking", [True, False]),
        "max_iter": hp.choice("max_iter", range(500, 2000, 50)),
    }

    def reg_objective(hyperparams):
        clf = SVR()
        clf.set_params(**hyperparams)
        model = clf.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = np.sqrt(((pred - y_val) ** 2).mean())
        if np.isnan(score):
            score = 1e8
        return {'loss': score, 'status': STATUS_OK}


    def clf_objective(hyperparams):
        clf = SVC() 
        clf.set_params(**hyperparams)
        model = clf.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = -roc_auc_score(y_val, pred)
        if np.isnan(score):
            score = 1e8
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    if config.is_classification():
        space = clf_space
        obj = clf_objective
    else:
        space = reg_space
        obj = reg_objective

    best = hyperopt.fmin(fn=obj, space=space, trials=trials, \
            algo=tpe.suggest, max_evals=max_trials, verbose=1, \
            rstate=np.random.RandomState(1))
    best_hyperparams = space_eval(space, best)
    
    best_loss = trials.best_trial['result']['loss']
    log("{:0.8f} {}".format(best_loss, best_hyperparams))
    
    _history = []
    for i, sc in enumerate(trials.losses()):
        p = space_eval(space, {k: v[i] for k, v in trials.vals.items()})
        _history.append((sc, {**p, **params}))
    history = sorted(_history, key = lambda x: x[0], reverse = False)
    for i, p in history:
        print("score: {0}, paramaters: {1}".format(i, p))
    return history 


