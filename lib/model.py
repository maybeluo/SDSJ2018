import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict

from lib.util import *
from lib.model_xgb import train_xgboost, hyperopt_xgboost, predict_xgboost 
from lib.model_lgb import train_lightgbm, hyperopt_lightgbm, predict_lightgbm 
from lib.model_svm import train_svm, hyperopt_svm, predict_svm

@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    if "leak" in config:
        return
   
    scaler = MinMaxScaler() # StandardScaler()
    scaler.fit(X.values)
    X = pd.DataFrame(scaler.transform(X.values), columns = X.columns)
    config["standscaler"] = scaler

    X_sample, y_sample = data_sample(X, y, nrows = 5000)

    history_lgb = hyperopt_lightgbm(X_sample, y_sample, config, max_trials = 50)
    best_lgb_loss, best_lgb_param = history_lgb[0]
    # train lgb
    config["model"]["lgb"] = []
    cv_lgb = train_lightgbm(X, y, config, best_lgb_param)
    config.save()
    print("[lightgbm] cv = {}".format(np.mean(cv_lgb)))
    
    #svm
    if (config["mode"] == "regression") and (config.time_left() >= config["time_limit"] * 0.7):
        history_svm = hyperopt_svm(X_sample, y_sample, config, max_trials = 50)
        best_svm_loss, best_svm_param = history_svm[0]
        config["model"]["svm"] = []
        cv_svm = train_svm(X, y, config, best_svm_param)
        print("[svm] cv = {}".format(np.mean(cv_svm)))
        
        if result_comparable(cv_lgb, cv_svm, config["mode"], 0.98) is True:
            config["ensemble"]["lgb"] = 0.5
            config["ensemble"]["svm"] = 0.5
            if 'holiday_detect' in config:
                config["ensemble"]["lgb"] = 0.2
                config["ensemble"]["svm"] = 0.8
            config.save()
        else:
            config["model"].pop("svm")

    if (config.time_left() >= config["time_limit"] * 0.8) and ('holiday_detect' not in config):
        history_xgb = hyperopt_xgboost(X_sample, y_sample, config, max_trials = 50)
        best_xgb_loss, best_xgb_param = history_xgb[0]
        config["model"]["xgb"] = []
        cv_xgb = train_xgboost(X, y, config, best_xgb_param)
        config.save()
        print("[xgboost] cv = {}".format(cv_xgb))
        
        if result_comparable(cv_lgb, cv_xgb, config["mode"], 0.98) is True:
            if "svm" in config["model"]:
                config["ensemble"]["lgb"] = 0.4
                config["ensemble"]["svm"] = 0.4
                config["ensemble"]["xgb"] = 0.2
            else:
                config["ensemble"]["lgb"] = 0.6
                config["ensemble"]["xgb"] = 0.4
            config.save()
        else:
            config["model"].pop("xgb")
    print("[train] used models: {}\n".format(config["model"].keys()))


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    if "leak" in config:
        preds = predict_leak(X, config)
        return preds
    else:
        if "standscaler" in config:
            scaler = config["standscaler"]
            X = pd.DataFrame(scaler.transform(X.values), columns = X.columns)
        
        n = 0
        pred_dic = {}
        for name in config["ensemble"]:
            w = config["ensemble"][name]
            print("Use model **{0}** for ensemble with weight {1}".format(name, w))
            pred_dic[name] = predict_by_name(X, config, name)
            n = len(pred_dic[name])

        preds = [0.0 for i in range(n)]
        for name in pred_dic:
            w = config["ensemble"][name]
            for i in range(n):
                preds[i] += pred_dic[name][i] * w
        return preds


def predict_by_name(X: pd.DataFrame, config: Config, name: str) -> List:
    if name == "xgb":
        return predict_xgboost(X, config)
    elif name == "lgb":
        return predict_lightgbm(X, config)
    elif name == "svm":
        return predict_svm(X, config)
    else:
        pass

@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def predict_leak(X: pd.DataFrame, config: Config) -> List:
    preds = pd.Series(0, index=X.index)

    for name, group in X.groupby(by=config["leak"]["id_col"]):
        gr = group.sort_values(config["leak"]["dt_col"])
        preds.loc[gr.index] = gr[config["leak"]["num_col"]].shift(config["leak"]["lag"])

    return preds.fillna(0).tolist()


