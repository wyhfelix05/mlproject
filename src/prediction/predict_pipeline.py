# src/prediction/predict_pipeline.py

import sys
import pandas as pd
from src.prediction.input_cleaning import PredictInputCleaner
from src.prediction.predictor import ModelPredictor
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    """
    工业化预测全流程：
    - 输入 dict / DataFrame
    - 自动清洗
    - 自动特征转换
    - 自动预测
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            self.cleaner = PredictInputCleaner()
            self.predictor = ModelPredictor(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                cleaner=self.cleaner
            )
            logging.info("PredictPipeline initialized.")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, input_data):
        """
        input_data: dict 或 DataFrame
        """
        try:
            logging.info("Starting prediction pipeline...")

            # 直接让 ModelPredictor 处理（清洗 → transform → predict）
            predictions = self.predictor.predict(input_data)
            logging.info("Prediction pipeline finished.")

            return predictions

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

        pipeline = PredictPipeline(
            model_path="artifacts/model.pkl",
            preprocessor_path="artifacts/preprocessor.pkl"
        )

        preds = pipeline.run(sample_json)
        print("Predicted values:", preds)

    except Exception as e:
        raise CustomException(e, sys)
