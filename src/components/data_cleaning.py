# src/data/clean_data.py

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataCleaningConfig:
    train_cleaned_path: str = os.path.join('artifacts', 'train_cleaned.csv')
    test_cleaned_path: str = os.path.join('artifacts', 'test_cleaned.csv')


class DataCleaner:
    """
    数据清洗类：根据 EDA 规则处理 Airbnb 数据
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------
    # 删除不必要的列
    # -------------------------
    def drop_unnecessary_columns(self):
        drop_cols = [
            'id', 'NAME', 'host id', 'host name',
            'country', 'country code', 'house_rules', 'license'
        ]
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # -------------------------
    # 金钱列处理
    # -------------------------
    def clean_money_columns(self):
        money_cols = ['price', 'service fee']
        for col in money_cols:
            if col in self.df.columns:
                s = self.df[col].astype(str).str.strip()
                mask_valid = s.str.startswith('$') | s.str.startswith('＄')
                s[~mask_valid] = np.nan
                s = s.str.replace(r'^[\$＄]\s*', '', regex=True).str.replace(',', '', regex=False)
                self.df[col] = pd.to_numeric(s, errors='coerce')
        if 'price' in self.df.columns:
            self.df = self.df.dropna(subset=['price'])

    # -------------------------
    # 整数列处理
    # -------------------------
    def clean_integer_columns(self):
        integer_cols = [
            'minimum nights', 'number of reviews', 'calculated host listings count'
        ]
        for col in integer_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df.loc[self.df[col] % 1 != 0, col] = np.nan
                self.df.loc[self.df[col] < 0, col] = np.nan
                self.df[col] = self.df[col].astype('Int64')

    # -------------------------
    # 浮点列处理
    # -------------------------
    def clean_float_columns(self):
        float_cols = ['reviews per month']
        for col in float_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df.loc[self.df[col] < 0, col] = np.nan

    # -------------------------
    # 类别变量处理（review rate 和 Construction year）
    # -------------------------
    def transform_categorical_columns(self):
        if 'review rate number' in self.df.columns:
            self.df['review rate number'] = pd.to_numeric(self.df['review rate number'], errors='coerce')
            self.df['review rate number'] = self.df['review rate number'].clip(1, 5)

        if 'Construction year' in self.df.columns:
            self.df['Construction year'] = pd.to_numeric(self.df['Construction year'], errors='coerce')
            self.df.loc[~self.df['Construction year'].between(1900, 2025), 'Construction year'] = np.nan
            self.df['Construction year'] = self.df['Construction year'].astype('Int64').astype(str)

    # -------------------------
    # 数值列截断（99%分位数）
    # -------------------------
    def truncate_numerical_columns(self):
        numerical_cols = [
            'minimum nights', 'number of reviews', 'reviews per month', 'calculated host listings count'
        ]

        if 'minimum nights' in self.df.columns:
            self.df = self.df[self.df['minimum nights'] > 0]

        for col in numerical_cols:
            if col in self.df.columns:
                upper = self.df[col].quantile(0.99)

                # 如果是整数列，取上整并保持 Int64
                if pd.api.types.is_integer_dtype(self.df[col]):
                    upper = int(np.ceil(upper))

                # mask 替换大于 upper 的值，兼容 pd.NA
                self.df[col] = self.df[col].mask(self.df[col] > upper, upper)

    # -------------------------
    # 特殊列处理
    # -------------------------
    def process_special_columns(self):
        if 'last review' in self.df.columns:
            reference_date = pd.Timestamp('2026-01-01')
            
            # 转换为 datetime，无法解析的设为 NaT
            last_review_dt = pd.to_datetime(self.df['last review'], errors='coerce')
            
            # 识别合理日期（小于 reference_date）
            valid_mask = last_review_dt < reference_date
            
            # 不合理的日期设为 NaT
            last_review_dt = last_review_dt.where(valid_mask, pd.NaT)
            
            # 转成天数差，并用可空整数类型存储
            self.df['last review'] = (reference_date - last_review_dt).dt.days.astype('Int64')

        if 'availability 365' in self.df.columns:
            self.df['availability 365'] = self.df['availability 365'].clip(0, 366)

        if 'neighbourhood' in self.df.columns:
            counts = self.df['neighbourhood'].value_counts()
            small_cats = counts[counts < 1000].index
            self.df['neighbourhood'] = self.df['neighbourhood'].replace(small_cats, 'Other')

        if 'neighbourhood group' in self.df.columns:
            allowed = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
            self.df = self.df[self.df['neighbourhood group'].isin(allowed)]

    # -------------------------
    # 全部清洗
    # -------------------------
    def clean_all(self):
        try:
            self.drop_unnecessary_columns()
            self.clean_money_columns()
            self.clean_integer_columns()
            self.clean_float_columns()
            self.transform_categorical_columns()
            self.truncate_numerical_columns()
            self.process_special_columns()
            logging.info("All data cleaning steps completed")
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------
    # 保存清洗后的数据
    # -------------------------
    def save_cleaned_data(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(path, index=False)

# -------------------------
# 主入口：分别清洗训练集和测试集
# -------------------------
if __name__ == "__main__":
    try:
        config = DataCleaningConfig()

        # 清洗训练集
        train_path = os.path.join('artifacts', 'train.csv')
        train_df = pd.read_csv(train_path, low_memory=False)
        cleaner_train = DataCleaner(train_df)
        cleaner_train.clean_all()
        cleaner_train.save_cleaned_data(config.train_cleaned_path)

        # 清洗测试集
        test_path = os.path.join('artifacts', 'test.csv')
        test_df = pd.read_csv(test_path, low_memory=False)
        cleaner_test = DataCleaner(test_df)
        cleaner_test.clean_all()
        cleaner_test.save_cleaned_data(config.test_cleaned_path)

    except Exception as e:
        raise CustomException(e, sys)
