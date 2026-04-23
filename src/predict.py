import time
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from src.utils.features_extractor import FeatureExtractor

class RevenuePredictor:
    """売上レンジ推定を行う推論パイプラインクラス"""
    
    def __init__(self, model_dir: str = "models/revenue_model"):
        self.model_dir = model_dir
        self.range_map = {
            "S": "売り上げ1兆円以上",
            "A": "売り上げ5000億円以上1兆円未満",
            "B": "売り上げ1000億円以上5000億円未満",
            "C": "売り上げ1000億円未満",
        }
        self._load_assets()

    def _load_assets(self):
        """モデルやスケーラーなどのアセットをロードする"""
        print("Loading AI models and scalers...")
        try:
            # SavedModel 形式でモデルをロード
            saved_model_path = os.path.join(self.model_dir, 'tf_model')
            self.model = tf.keras.models.load_model(saved_model_path)
            
            # ピクルファイルからスケーラーとラベルエンコーダーをロード
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.le = pickle.load(f)
            
            # モデル設定をロード
            with open(os.path.join(self.model_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self.extractor = FeatureExtractor()
            print(f"✓ Model loaded: {self.config.get('model_name', 'Unknown')}")
            print(f"✓ Classes: {self.config.get('classes', [])}")
        except Exception as e:
            print(f"Error loading assets: {e}")
            raise RuntimeError("Failed to load models. Check if 'models/' directory is correctly mounted.")

    def predict(self, url: str) -> Tuple[Dict[str, Any], float]:
        """
        URLを受け取り、推論結果の辞書と処理時間を返す
        """
        start_time = time.time()

        # Step 1: Scrapyによるクローリングと特徴量作成
        features, summary = self.extractor.extract_from_url(url)
        
        if features is None:
            # API層でHTTPExceptionに変換できるようにValueErrorを投げる
            raise ValueError("Crawl failed or timed out.")

        # Step 2: 推論（マルチ入力に合わせてデータを整形）
        X_text = features['text'].reshape(1, -1)
        X_struct = self.scaler.transform(features['struct'].reshape(1, -1))

        preds_prob = self.model.predict(
            {"text_vectors": X_text, "structural_features": X_struct},
            verbose=0
        )[0]

        # Step 3: 結果整形
        class_idx = np.argmax(preds_prob)
        predicted_class = self.le.classes_[class_idx]
        
        prob_dict = {
            str(cls): round(float(prob), 4) 
            for cls, prob in zip(self.le.classes_, preds_prob)
        }

        result = {
            "estimated_revenue_class": predicted_class,
            "estimated_revenue_range": self.range_map.get(predicted_class, "不明"),
            "confidence": round(float(np.max(preds_prob)), 4),
            "class_probabilities": prob_dict,
            "features_summary": summary,
        }

        processing_time_sec = round(time.time() - start_time, 2)
        
        return result, processing_time_sec