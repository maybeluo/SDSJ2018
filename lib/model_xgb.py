import time
import pandas as pd
import numpy as np
import xgboost as xgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import *
from typing import Tuple, List, Dict


@timeit
def train_xgboost(X: pd.DataFrame, y: pd.Series, config: Config, params: Dict):
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
        dtrain = xgb.DMatrix(X_train, label = y_train)
        dvalid = xgb.DMatrix(X_valid, label = y_valid)
        watchlist = [(dvalid, 'valid')]

        cur_model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50, verbose_eval=100)
        cv_res.append(cur_model.best_score)
        config["model"]["xgb"].append(cur_model)
        
        fe_time = time.time() 
        if fe_time - fb_time >= 240: # 4min
            break
        if (fe_time - stime) / (fid + 1) * 2.0 >= config.time_left():
            break
    return np.mean(cv_res)

@timeit
def predict_xgboost(X: pd.DataFrame, config: Config) -> List:
    ans = []
    dtest = xgb.DMatrix(X)
    for model in config["model"]["xgb"]:
        p = model.predict(dtest)
        if config["non_negative_target"]:
            p = [max(0, i) for i in p]
        ans.append(p)
    return np.mean(ans, axis = 0).flatten()


@timeit
def hyperopt_xgboost(X: pd.DataFrame, y: pd.Series, config: Config, max_trials: int=50):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    params = {
        "booster": "gbtree",
        "objective": "reg:linear" if config["mode"] == "regression" else "binary:logistic",
        "eval_metric": "rmse" if config["mode"] == "regression" else "auc",
        "silent": True,
        "seed": 1,
    }

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.08), np.log(0.2)),
        "max_depth": hp.choice("max_depth", [2, 3, 4, 5, 6]),
        "max_leaves": hp.choice("max_leaves", np.linspace(10, 200, 50, dtype=int)),
        "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.6, 1.0, 0.1),
        "colsample_bylevel": hp.quniform("colsample_bylevel", 0.6, 1.0, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    watchlist = [(dvalid, 'valid')]
    def objective(hyperparams):
        model = xgb.train({**params, **hyperparams}, dtrain, 300, watchlist,
                early_stopping_rounds=100, verbose_eval=100)

        score = model.best_score
        if config.is_classification():
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, \
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


