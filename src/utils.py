import os
import sys
import pickle
import pandas as pd

from src.exception import CustomException


def ensure_dir(path: str):
    """
    如果目录不存在则自动创建。
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """
    使用 pickle 将对象保存到指定路径。
    """
    try:
        dir_path = os.path.dirname(file_path)
        ensure_dir(dir_path)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    从指定路径加载 pickle 对象。
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_csv(file_path: str) -> pd.DataFrame:
    """
    安全加载 CSV 文件为 DataFrame。
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV 文件不存在: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def save_csv(df: pd.DataFrame, file_path: str, index=False):
    """
    安全保存 DataFrame 为 CSV。
    """
    try:
        dir_path = os.path.dirname(file_path)
        ensure_dir(dir_path)

        df.to_csv(file_path, index=index)
    except Exception as e:
        raise CustomException(e, sys)
