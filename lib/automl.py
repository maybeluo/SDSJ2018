import os
import pandas as pd
import numpy as np
from lib.util import timeit, Config
from lib.read import read_df
from lib.preprocess import preprocess, holiday_detect
from lib.model import train, predict, validate
from typing import Optional


class AutoML:
    def __init__(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.config = Config(model_dir)

    def train(self, train_csv: str, mode: str):
        self.config["task"] = "train"
        self.config["mode"] = mode
        self.config["model"] = {}
        self.config["ensemble"] = {"lgb": 1}

        self.config.tmp_dir = self.config.model_dir + "/tmp"
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        # load holiday
        path_holiday = './holiday.csv'
        holiday = pd.read_csv(path_holiday, \
                      encoding='utf-8', low_memory=False, dtype={'holiday':str})['holiday'].values
        self.config['holiday'] = set(holiday)

        df = read_df(train_csv, self.config)
        print(df.shape)

        holiday_detect(df, self.config)

        preprocess(df, self.config)

        y = df["target"]
        X = df.drop("target", axis=1)

        train(X, y, self.config)

    def predict(self, test_csv: str, prediction_csv: str) -> (pd.DataFrame, Optional[np.float64]):
        self.config["task"] = "predict"
        self.config.tmp_dir = os.path.dirname(prediction_csv) + "/tmp"
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        result = {
            "line_id": [],
            "prediction": []
        }
        if 'holiday_detect' in self.config:
            result["datetime"] = []

        for X in pd.read_csv(
                test_csv,
                encoding="utf-8",
                low_memory=False,
                dtype=self.config["dtype"],
                parse_dates=self.config["parse_dates"],
                chunksize=self.config["nrows"]
        ):
            result["line_id"] += list(X["line_id"])
            if 'holiday_detect' in self.config:
                dt_fea = self.config['holiday_detect']
                result["datetime"] += list(X[dt_fea])

            preprocess(X, self.config)
            result["prediction"] += list(predict(X, self.config))

        result = pd.DataFrame(result)
        
        # post process for holiday
        if 'holiday_detect' in self.config:
            holiday = self.config['holiday']
            for idx, row in result.iterrows():
                dt =  row['datetime']
                dt_str = str(dt).split(' ')[0].strip()
                if dt_str in holiday or dt.weekday() == 5 or dt.weekday() == 6:
                    result.loc[idx, 'prediction'] = 0
            
            result.drop(["datetime"], axis = 1, inplace=True)

        result.to_csv(prediction_csv, index=False)
        
        target_csv = test_csv.replace("test", "test-target")
        if os.path.exists(target_csv):
            score = validate(result, target_csv, self.config["mode"])
        else:
            score = None

        return result, score

    @timeit
    def save(self):
        self.config.save()

    @timeit
    def load(self):
        self.config.load()
