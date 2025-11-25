import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns =  [
                'lat',
                'long',
                'service fee',
                'minimum nights',
                'number of reviews',
                'last review',
                'reviews per month',
                'calculated host listings count',
                'availability 365',
            ]
            categorical_columns = [
                'host_identity_verified',
                'neighbourhood group',
                'neighbourhood',
                'instant_bookable',
                'cancellation_policy',
                'room type',
                'Construction year',
                'review rate number',
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ],
                sparse_threshold=0

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"
            numerical_columns =  [
                'lat',
                'long',
                'service fee',
                'minimum nights',
                'number of reviews',
                'last review',
                'reviews per month',
                'calculated host listings count',
                'availability 365',
            ]


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
# -------------------------
# 主入口：数据转换
# -------------------------
if __name__ == "__main__":
    from src.components.clean_data import DataCleaningConfig

    try:
        # 清洗后的训练集和测试集路径
        config = DataCleaningConfig()
        train_cleaned_path = config.train_cleaned_path
        test_cleaned_path = config.test_cleaned_path

        # 实例化 DataTransformation
        data_transformation = DataTransformation()

        # 执行数据转换
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_cleaned_path,
            test_path=test_cleaned_path
        )

        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
        logging.info(f"Transformed train array shape: {train_arr.shape}")
        logging.info(f"Transformed test array shape: {test_arr.shape}")

    except Exception as e:
        raise CustomException(e, sys)
