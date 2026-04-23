from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any

# 推論パイプラインクラスをインポート
from src.predict import RevenuePredictor

app = FastAPI(title="Revenue Range Estimation API")

# ==========================================
# 1. モデルと資産のロード（起動時に1回）
# ==========================================
# インスタンス化することで、内部の __init__ が走りモデルがメモリにロードされます
try:
    predictor = RevenuePredictor(model_dir="models/revenue_model")
except RuntimeError as e:
    # 起動時にモデルが読み込めない場合はフェイルファストさせる
    raise e

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
    url_str = str(request.url)

    try:
        # predict.py に記述した推論処理を呼び出す
        result_dict, processing_time = predictor.predict(url_str)
        
    except ValueError as e:
        # predict側でクローリング失敗時などに投げたValueErrorをキャッチして500エラーにする
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # その他の予期せぬエラーのハンドリング
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

    # レスポンスの構築（辞書を展開して渡す）
    return EstimateResponse(
        url=url_str,
        **result_dict,
        processing_time_sec=processing_time
    )