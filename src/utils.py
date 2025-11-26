import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    高效版 evaluate_models：每个模型只训练一次，兼容原主程序使用。

    参数:
        X_train, y_train: 训练集
        X_test, y_test: 测试集
        models: 字典 {模型名: 模型对象}，函数内部会直接覆盖训练好的模型
        param: 字典 {模型名: 参数网格}

    返回:
        report: {模型名: 测试集 r2_score}
        （原 models 字典已更新为训练好的模型）
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            # 如果有参数网格则用 GridSearchCV 找最佳参数并训练
            if para:
                para_checked = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in para.items()}
                gs = GridSearchCV(model, para_checked, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)

                # 使用 GridSearchCV 返回的训练好的最佳模型
                best_model = gs.best_estimator_
            else:
                # 没有参数网格直接 fit 原始模型
                best_model = model
                best_model.fit(X_train, y_train)

            # 覆盖原 models 字典里的对象
            models[model_name] = best_model

            # 预测并计算 r2_score
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report  # 仅返回 report，主程序不用改

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)