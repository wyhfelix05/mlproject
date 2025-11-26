# src/pipeline.py

import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaningConfig, DataCleaner
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_pipeline():
    try:
        logging.info("===== PIPELINE STARTED =====")

        # ----------------------------------------------------
        # 1. Data Ingestion
        # ----------------------------------------------------
        logging.info("Step 1: Data Ingestion")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info(f"Raw train path: {train_path}")
        logging.info(f"Raw test path:  {test_path}")

        # ----------------------------------------------------
        # 2. Data Cleaning
        # ----------------------------------------------------
        logging.info("Step 2: Data Cleaning")

        cleaning_config = DataCleaningConfig()

        # 读取原始 train / test
        import pandas as pd
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)

        cleaner_train = DataCleaner(train_df)
        cleaner_train.clean_all()
        cleaner_train.save_cleaned_data(cleaning_config.train_cleaned_path)

        cleaner_test = DataCleaner(test_df)
        cleaner_test.clean_all()
        cleaner_test.save_cleaned_data(cleaning_config.test_cleaned_path)

        logging.info(f"Cleaned train saved: {cleaning_config.train_cleaned_path}")
        logging.info(f"Cleaned test saved:  {cleaning_config.test_cleaned_path}")

        # ----------------------------------------------------
        # 3. Data Transformation
        # ----------------------------------------------------
        logging.info("Step 3: Data Transformation")

        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path=cleaning_config.train_cleaned_path,
            test_path=cleaning_config.test_cleaned_path
        )

        logging.info(f"Preprocessor saved at: {preprocessor_path}")
        logging.info(f"Train array shape: {train_arr.shape}")
        logging.info(f"Test array shape:  {test_arr.shape}")

        # ----------------------------------------------------
        # 4. Model Training
        # ----------------------------------------------------
        logging.info("Step 4: Model Training")

        trainer = ModelTrainer()
        results = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Best model: {results['best_model_name']}")
        logging.info(f"Best model validation score: {results['best_model_score']}")
        logging.info(f"Best model test R²: {results['best_model_r2']}")


        logging.info("===== PIPELINE FINISHED SUCCESSFULLY =====")

        print("\n=== PIPELINE RESULT ===")
        print("Best model:", results["best_model_name"])
        print("Training CV score:", results["best_model_score"])
        print("Test R²:", results["best_model_r2"])
        print(f"预处理器位置：{preprocessor_path}")
        print(f"模型保存位置：artifacts/model.pkl")



    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()
