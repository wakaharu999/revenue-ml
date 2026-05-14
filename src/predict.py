import os
import time
import json
import tempfile
import numpy as np
import torch
from multiprocessing import Process
from transformers import AutoTokenizer
from scrapy.crawler import CrawlerProcess
from typing import Dict, Any, Tuple

# 自作モジュールのインポート
from src.model import HierarchicalAttentionBERT
from src.crawler import RevenueSpider, CATEGORIES

def run_spider(url, temp_file):
    """Scrapyを別プロセスで実行するためのラッパー関数"""
    process = CrawlerProcess()
    process.crawl(RevenueSpider, start_url=url, temp_file=temp_file)
    process.start()

class RevenuePredictor:
    """階層型BERTを用いた売上レンジ推定推論パイプライン"""
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        # READMEの定義に合わせた売上レンジ
        self.range_map = {
            "S": "2兆円以上",
            "A": "8000億円〜2兆円未満",
            "B": "5000億円〜8000億円未満",
            "C": "〜5000億円未満",
            "D": "スタートアップ企業",
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for inference: {self.device}")
        
        self.model_name = "cl-tohoku/bert-base-japanese-v3"
        self._load_assets()

    def _load_assets(self):
        """トークナイザ、マッピング辞書、モデル重みのロード"""
        print("Loading PyTorch model and tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # マッピングの読み込み
            mapping_path = os.path.join(self.data_dir, "label_mappings.json")
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            self.revenue2id = mappings["revenue2id"]
            self.category2id = mappings["category2id"]
            self.id2revenue = {int(v): k for k, v in self.revenue2id.items()}
            
            # モデルの初期化と重みのロード
            self.model = HierarchicalAttentionBERT(
                model_name=self.model_name,
                num_categories=len(self.category2id),
                num_classes=len(self.revenue2id)
            )
            
            model_path = os.path.join(self.model_dir, "model.pt")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval() # 推論モード
            
            print("✓ PyTorch Model successfully loaded.")
            
        except Exception as e:
            print(f"Error loading assets: {e}")
            raise RuntimeError("モデルのロードに失敗しました。model.pt や label_mappings.json の配置を確認してください。")

    def _preprocess_texts(self, collected_texts: Dict[str, str], max_chunks=4, max_len=512):
        """クローリングしたテキストをBERTの入力テンソルに変換する"""
        num_categories = len(self.category2id)
        
        # テンソルの初期化
        input_ids = torch.zeros((1, num_categories, max_chunks, max_len), dtype=torch.long)
        attention_mask = torch.zeros((1, num_categories, max_chunks, max_len), dtype=torch.long)
        chunk_mask = torch.zeros((1, num_categories, max_chunks), dtype=torch.float)
        category_ids_tensor = torch.zeros((1, num_categories), dtype=torch.long)
        
        for cat_name, text in collected_texts.items():
            if cat_name not in self.category2id:
                continue
                
            cat_idx = self.category2id[cat_name]
            category_ids_tensor[0, cat_idx] = cat_idx
            
            if not text.strip():
                continue # テキストが無い場合はゼロ埋めのまま
                
            # 文字列をざっくりチャンク分割（1チャンク約400文字目安）
            chunks_text = [text[i:i+400] for i in range(0, len(text), 400)][:max_chunks]
            
            for chunk_idx, chunk_str in enumerate(chunks_text):
                encoded = self.tokenizer(
                    chunk_str,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids[0, cat_idx, chunk_idx] = encoded['input_ids'][0]
                attention_mask[0, cat_idx, chunk_idx] = encoded['attention_mask'][0]
                chunk_mask[0, cat_idx, chunk_idx] = 1.0 # 有効なチャンクとしてフラグを立てる
                
        return input_ids.to(self.device), attention_mask.to(self.device), chunk_mask.to(self.device), category_ids_tensor.to(self.device)

    def predict(self, url: str) -> Tuple[Dict[str, Any], float]:
        """URLを受け取り、推論結果の辞書と処理時間を返す"""
        start_time = time.time()

        # ----------------------------------------
        # Step 1: Scrapyによるクローリング
        # ----------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            temp_file_path = tmp.name

        p = Process(target=run_spider, args=(url, temp_file_path))
        p.start()
        p.join(timeout=60) # 余裕を持って60秒
        
        if p.is_alive():
            p.terminate()
            p.join()
            os.remove(temp_file_path)
            raise ValueError("クローリングがタイムアウトしました。")

        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                crawl_result = json.load(f)
        except Exception:
            os.remove(temp_file_path)
            raise ValueError("クローリングに失敗したか、有効なデータが取得できませんでした。")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        collected_texts = crawl_result.get('texts', {})
        summary = crawl_result.get('summary', {})

        # ----------------------------------------
        # Step 2: 前処理と推論
        # ----------------------------------------
        input_ids, att_mask, c_mask, cat_ids = self._preprocess_texts(collected_texts)

        with torch.no_grad():
            # 自動混合精度 (AMP) で推論速度を最適化
            if self.device.type == 'cuda':
                with torch.autocast(device_type='cuda'):
                    logits = self.model(input_ids, att_mask, c_mask, cat_ids)
            else:
                logits = self.model(input_ids, att_mask, c_mask, cat_ids)
                
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # ----------------------------------------
        # Step 3: 結果整形
        # ----------------------------------------
        class_idx = int(np.argmax(probs))
        predicted_class = self.id2revenue[class_idx]
        
        # 確率辞書の作成
        prob_dict = {
            self.id2revenue[i]: round(float(probs[i]), 4) 
            for i in range(len(probs))
        }
        # S, A, B, C, D の順に見やすくソート
        prob_dict = {k: prob_dict[k] for k in ["S", "A", "B", "C", "D"] if k in prob_dict}

        result = {
            "estimated_revenue_class": predicted_class,
            "estimated_revenue_range": self.range_map.get(predicted_class, "不明"),
            "confidence": round(float(np.max(probs)), 4),
            "class_probabilities": prob_dict,
            "features_summary": summary,
        }

        processing_time_sec = round(time.time() - start_time, 2)
        
        return result, processing_time_sec