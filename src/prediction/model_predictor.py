# src/prediction/model_predictor.py

import sys
import pandas as pd
import numpy as np
from src.utils import load_object
from src.prediction.input_cleaning import PredictInputCleaner
from src.exception import CustomException
from src.logger import logging


class ModelPredictor:
    """
    工业化推荐版本：
    - 支持依赖注入清洗器（PredictInputCleaner）
    - transform/predict 分离，职责清晰
    - 支持单行 dict 或 pd.DataFrame 输入
    """

    def __init__(self, model_path: str, preprocessor_path: str, cleaner=None):
        """
        Args:
            model_path: 训练好的模型 pickle 文件路径
            preprocessor_path: 训练时保存的 preprocessor pickle 文件路径
            cleaner: 可选的自定义清洗器（默认 PredictInputCleaner）
        """
        try:
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            self.cleaner = cleaner  # 外部注入清洗器
            if self.cleaner is None:
                from src.prediction.input_cleaning import PredictInputCleaner
                self.cleaner = PredictInputCleaner()

            logging.info(f"Loaded model from {model_path}")
            logging.info(f"Loaded preprocessor from {preprocessor_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        对清洗后的 DataFrame 做特征转换
        """
        try:
            return self.preprocessor.transform(df)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_features(self, X: np.ndarray) -> list:
        """
        对特征矩阵做预测
        """
        try:
            preds = self.model.predict(X)
            return preds.tolist()
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data) -> list:
        """
        完整预测流程：dict 或 DataFrame
        """
        try:
            # 1. 清洗
            if isinstance(input_data, dict):
                df_cleaned = self.cleaner.clean(input_data)
            elif isinstance(input_data, pd.DataFrame):
                df_cleaned = input_data
            else:
                raise ValueError("input_data must be dict or pandas DataFrame")

            # 2. 特征转换
            X = self.transform(df_cleaned)

            # 3. 预测
            return self.predict_from_features(X)

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# 本地调试入口
# -----------------------------
if __name__ == "__main__":
    try:
        sample_json = {
            "id": "29262675",
            "NAME": "Brand New Bright and Clean Private bedrooms",
            "host id": "30909368236",
            "host_identity_verified": "verified",
            "host name": "Alan",
            "neighbourhood group": "Brooklyn",
            "neighbourhood": "Williamsburg",
            "lat": "40.7128",
            "long": "-73.9653",
            "country": "United States",
            "country code": "US",
            "instant_bookable": "TRUE",
            "cancellation_policy": "flexible",
            "room type": "Entire home/apt",
            "Construction year": "2010",
            "service fee": "$100",
            "minimum nights": "5",
            "number of reviews": "100",
            "last review": "2022/10/19",
            "reviews per month": "0.8",
            "review rate number": "5",
            "calculated host listings count": "2",
            "availability 365": "200"
        }

        cleaner = PredictInputCleaner()  # 可以换成自定义清洗器
        predictor = ModelPredictor(
            model_path="artifacts/model.pkl",
            preprocessor_path="artifacts/preprocessor.pkl",
            cleaner=cleaner
        )

        # 完整预测
        preds = predictor.predict(sample_json)
        print("Predicted values:", preds)

    except Exception as e:
        raise CustomException(e, sys)
