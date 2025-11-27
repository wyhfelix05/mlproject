import json
import os
import pickle
import traceback
import pandas as pd

from src.prediction.input_cleaning import PredictInputCleaner

# ==============================
# 🔹 冷启动初始化（只执行一次）
# ==============================
try:
    # 模型和预处理器路径（放在 Lambda zip 中的 artifacts 文件夹里）
    MODEL_PATH = os.path.join("artifacts", "model.pkl")
    PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

    # 加载模型和预处理器
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 初始化预测输入清洗器
    cleaner = PredictInputCleaner()

    CONFIG = {
        "version": "1.0",
        "env": os.environ.get("STAGE", "dev")
    }

except Exception as e:
    INIT_ERROR = str(e)
    print("Initialization failed:", e)
    print(traceback.format_exc())
else:
    INIT_ERROR = None

# ==============================
# 🔹 Lambda 主 handler
# ==============================
def lambda_handler(event, context):
    """
    Lambda handler：接收 API Gateway / Lambda Function URL / 本地测试事件
    """
    try:
        # 1) 冷启动检查
        if INIT_ERROR:
            return _response(500, {
                "error": "Initialization failed",
                "detail": INIT_ERROR
            })

        # 2) 获取请求 body
        body = event.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return _response(400, {"error": "Invalid JSON body"})

        elif body is None:
            # 本地测试直接传 dict
            if isinstance(event, dict):
                body = event
            else:
                return _response(400, {"error": "Empty request body"})

        # ==============================
        # 🔹 推理逻辑
        # ==============================
        # 1️⃣ 清洗输入
        df_clean = cleaner.clean(body)

        # 2️⃣ 特征处理
        X = preprocessor.transform(df_clean)

        # 3️⃣ 模型预测
        pred = model.predict(X)

        # 4️⃣ 返回预测结果
        return _response(200, {"prediction": pred.tolist()})

    except Exception as e:
        print("Error during execution:", e)
        print(traceback.format_exc())
        return _response(500, {"error": str(e)})

# ==============================
# 🔹 统一返回格式
# ==============================
def _response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body, ensure_ascii=False)
    }

if __name__ == "__main__":
    # 本地测试事件
    test_event = {
        "body": {
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
    }

    # Lambda handler 测试
    response = lambda_handler(test_event, context=None)
    print("=== Lambda Handler Response ===")
    print(response)

    # 如果只想看预测结果
    body = json.loads(response["body"])
    print("\n=== Prediction Result ===")
    print(body.get("prediction"))
