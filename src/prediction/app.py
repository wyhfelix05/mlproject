from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os
from src.prediction.input_cleaning import PredictInputCleaner

# -------------------------
# 1️⃣ Pydantic 输入模型（Python 合法属性名 + alias 映射原始列名）
# -------------------------
class InputData(BaseModel):
    id: str = Field(..., alias="id")
    name: str = Field(..., alias="NAME")
    host_id: str = Field(None, alias="host id")
    host_identity_verified: str = Field(None, alias="host_identity_verified")
    host_name: str = Field(None, alias="host name")
    neighbourhood_group: str = Field(None, alias="neighbourhood group")
    neighbourhood: str = Field(None, alias="neighbourhood")
    lat: str = Field(None, alias="lat")
    long: str = Field(None, alias="long")
    country: str = Field(None, alias="country")
    country_code: str = Field(None, alias="country code")
    instant_bookable: str = Field(None, alias="instant_bookable")
    cancellation_policy: str = Field(None, alias="cancellation_policy")
    room_type: str = Field(None, alias="room type")
    construction_year: str = Field(None, alias="Construction year")
    service_fee: str = Field(None, alias="service fee")
    minimum_nights: str = Field(None, alias="minimum nights")
    number_of_reviews: str = Field(None, alias="number of reviews")
    last_review: str = Field(None, alias="last review")
    reviews_per_month: str = Field(None, alias="reviews per month")
    review_rate_number: str = Field(None, alias="review rate number")
    calculated_host_listings_count: str = Field(None, alias="calculated host listings count")
    availability_365: str = Field(None, alias="availability 365")
    house_rules: str = Field(None, alias="house_rules")
    license: str = Field(None, alias="license")

    class Config:
        validate_by_name = True  # 支持通过字段名或 alias 赋值

# -------------------------
# 2️⃣ 创建 FastAPI 应用
# -------------------------
app = FastAPI(title="ML Prediction API")

# -------------------------
# 3️⃣ 加载 pipeline 和模型
# -------------------------
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 初始化预测输入清洗器
cleaner = PredictInputCleaner()

# -------------------------
# 4️⃣ 根路由
# -------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the ML Prediction API"}

# -------------------------
# 5️⃣ 预测路由
# -------------------------
@app.post("/predict")
def predict(data: InputData):
    # 1️⃣ JSON → dict（保留 alias 对应原始列名）
    input_dict = data.dict(by_alias=True)

    # 2️⃣ 调用 PredictInputCleaner 进行清洗
    df_clean = cleaner.clean(input_dict)

    # 3️⃣ 调用 pipeline 进行特征转换
    X = preprocessor.transform(df_clean)

    # 4️⃣ 模型预测
    pred = model.predict(X)

    # 5️⃣ 返回预测结果
    return {"prediction": pred.tolist()}

