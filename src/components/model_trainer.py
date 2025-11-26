import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # -----------------------------
            # Define all candidate models
            # -----------------------------
            models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet(),
                "XGBRegressor": XGBRegressor(n_jobs=-1),
            }

            # -----------------------------
            # Parameter grids for hyperparameter tuning
            # -----------------------------
            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "Ridge Regression": {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]},
                "Lasso Regression": {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
                "ElasticNet Regression": {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                "XGBRegressor": {'learning_rate': [.1, .01, .05, .001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            # -----------------------------
            # Evaluate models and find best
            # -----------------------------
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model meets performance threshold (0.6)")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # -----------------------------
            # Save the best model
            # -----------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # -----------------------------
            # Evaluate final test R²
            # -----------------------------
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)

            return {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "best_model_r2": test_r2,
                "best_model": best_model
            }

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------
# 主入口：模型训练
# -------------------------
if __name__ == "__main__":
    try:
        from src.components.transform_data import DataTransformation
        from src.components.clean_data import DataCleaningConfig

        # 1. 获取清洗后数据的路径
        config = DataCleaningConfig()
        train_cleaned_path = config.train_cleaned_path
        test_cleaned_path = config.test_cleaned_path

        # 2. 执行数据转换
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_cleaned_path,
            test_path=test_cleaned_path
        )

        print(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")

        # 3. 执行模型训练
        trainer = ModelTrainer()
        results = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"\n最佳模型: {results['best_model_name']}")
        print(f"验证集最佳分数: {results['best_model_score']:.4f}")
        print(f"测试集 R²: {results['best_model_r2']:.4f}")

    except Exception as e:
        print("运行 ModelTrainer 的 main 时发生错误：", e)
