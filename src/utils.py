import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """Train and evaluate multiple models.

    Returns a dict mapping model name -> r2 score on the test set.
    """
    try:
        report = {}
        for name, model in models.items():
            params = param.get(name, {})
            if params:
                gs = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[name] = score

        return report
    except Exception as e:
        raise CustomException(e, sys)