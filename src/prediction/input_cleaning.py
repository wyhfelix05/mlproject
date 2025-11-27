# src/prediction/input_cleaning.py

import sys
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

from src.logger import logging
from src.exception import CustomException


class PredictInputCleaner:
    """
    预测阶段专用清洗（工业化版本）：
    - 仅做格式化、补列、类型转换
    - 不做过滤、不删列、不做 outlier、不做截断
    """

    def __init__(self):
        try:
            self.expected_columns = [
                # 数值列
                'lat',
                'long',
                'service fee',
                'minimum nights',
                'number of reviews',
                'last review',
                'reviews per month',
                'calculated host listings count',
                'availability 365',
                # 类别列（原始列名，之后 OneHotEncoder 会扩展）
                'host_identity_verified',
                'neighbourhood group',
                'neighbourhood',
                'instant_bookable',
                'cancellation_policy',
                'room type',
                'Construction year',
                'review rate number'
            ]

            logging.info("PredictInputCleaner initialized with expected columns.")

        except Exception as e:
            raise CustomException(e, sys)

    def clean(self, input_dict: dict) -> pd.DataFrame:
        """
        输入 dict → 输出干净 DataFrame（列顺序对齐训练特征）
        """
        try:
            logging.info("Starting input cleaning...")

            # 1. JSON → DataFrame
            df = pd.DataFrame([input_dict])
            logging.info("Converted input dict to DataFrame.")

            # 2. 补齐训练列
            for col in self.expected_columns:
                if col not in df.columns:
                    df[col] = None
            logging.info("Missing expected columns filled with None.")

            # 3. 金钱列处理
            for money_col in ["service fee"]:
                if money_col in df.columns:
                    s = df[money_col].astype(str).str.strip()
                    mask_valid = s.str.startswith("$") | s.str.startswith("＄")
                    s = s.mask(~mask_valid)
                    s = s.str.replace(r'^[\$＄]\s*', '', regex=True).str.replace(',', '', regex=False)
                    df[money_col] = pd.to_numeric(s, errors='coerce')
            logging.info("Processed money columns.")

            # 4. 整数列（转 float）
            integer_cols = [
                "minimum nights", "number of reviews",
                "calculated host listings count", "availability 365"
            ]
            for col in integer_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            logging.info("Processed integer-like columns.")

            # 5. 浮点列
            float_cols = ["lat", "long", "reviews per month"]
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            logging.info("Processed float-like columns.")

            # 6. review rate
            if "review rate number" in df.columns:
                df["review rate number"] = pd.to_numeric(df["review rate number"], errors="coerce")

            # 7. Construction year
            if "Construction year" in df.columns:
                df["Construction year"] = pd.to_numeric(df["Construction year"], errors="coerce")

            # 8. instant_bookable 转 bool
            if "instant_bookable" in df.columns:
                df["instant_bookable"] = df["instant_bookable"].replace({"TRUE": True, "FALSE": False})

            # 9. last review → 天数差
            if "last review" in df.columns:
                reference_date = pd.Timestamp("2026-01-01")
                last_review_dt = pd.to_datetime(df["last review"], errors="coerce")
                valid_mask = last_review_dt < reference_date
                last_review_dt = last_review_dt.where(valid_mask, pd.NaT)
                df["last review"] = (reference_date - last_review_dt).dt.days
            logging.info("Processed last review date column.")

            # 10. neighbourhood
            if "neighbourhood" in df.columns:
                df["neighbourhood"] = df["neighbourhood"].str.strip()
                df["neighbourhood"] = df["neighbourhood"].fillna("Other")

            # 11. neighbourhood group
            if "neighbourhood group" in df.columns:
                allowed = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
                df.loc[~df["neighbourhood group"].isin(allowed), "neighbourhood group"] = None

            # 12. 重新排列列顺序
            df = df[self.expected_columns]
            logging.info("Reordered columns to match expected feature order.")

            logging.info("Input cleaning finished successfully.")
            return df

        except Exception as e:
            logging.error("Error occurred during input cleaning.")
            raise CustomException(e, sys)


# -----------------------------
# 可测试 main
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

        cleaner = PredictInputCleaner()
        df = cleaner.clean(sample_json)

        print("\n=== Cleaned DataFrame ===")
        print(df)

    except Exception as e:
        raise CustomException(e, sys)
