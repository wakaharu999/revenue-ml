import os
import time
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any

# 特徴量抽出ロジック（後述のextractor.py）
from src.utils.feature_extractor import FeatureExtractor

app = FastAPI(title="Revenue Range Estimation API")

# ==========================================
# 1. モデルと資産のロード（起動時に1回）
# ==========================================
# パスはDockerコンテナ内の構造に合わせる
MODEL_DIR = "models/revenue_model"
print("Loading AI models and scalers...")

try:
    model = tf.keras.models.load_model(f"{MODEL_DIR}/tf_model.keras")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    # 特徴量抽出器の初期化
    extractor = FeatureExtractor()
except Exception as e:
    print(f"Error loading assets: {e}")
    raise RuntimeError("Failed to load models. Check if 'models/' directory is correctly mounted.")

RANGE_MAP = {
    "S": "あ",
    "A": "い",
    "B": "う",
    "C": "え",
}

# ==========================================
# 2. スキーマ定義
# ==========================================
class EstimateRequest(BaseModel):
    url: HttpUrl

class EstimateResponse(BaseModel):
    url: str
    estimated_revenue_class: str
    estimated_revenue_range: str
    confidence: float
    class_probabilities: Dict[str, float]
    features_summary: Dict[str, Any]
    processing_time_sec: float

# ==========================================
# 3. エンドポイント
# ==========================================
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest):
    start_time = time.time()
    url_str = str(request.url)

    # Step 1: Scrapyによるクローリングと特徴量作成
    # 内部でmultiprocessingを使用してReactorエラーを回避
    features, summary = extractor.extract_from_url(url_str)
    
    if features is None:
        raise HTTPException(status_code=500, detail="Crawl failed or timed out.")

    # Step 2: 推論（マルチ入力に合わせてデータを整形）
    # 学習時と同じ次元数であることを確認
    X_text = features['text'].reshape(1, -1) # (1, 5376)
    X_struct = scaler.transform(features['struct'].reshape(1, -1)) # (1, 17)

    preds_prob = model.predict(
        {"text_vectors": X_text, "structural_features": X_struct},
        verbose=0
    )[0]

    # Step 3: 結果整形
    class_idx = np.argmax(preds_prob)
    predicted_class = le.classes_[class_idx]
    
    prob_dict = {
        str(cls): round(float(prob), 4) 
        for cls, prob in zip(le.classes_, preds_prob)
    }

    return {
        "url": url_str,
        "estimated_revenue_class": predicted_class,
        "estimated_revenue_range": RANGE_MAP.get(predicted_class, "不明"),
        "confidence": round(float(np.max(preds_prob)), 4),
        "class_probabilities": prob_dict,
        "features_summary": summary,
        "processing_time_sec": round(time.time() - start_time, 2)
    }