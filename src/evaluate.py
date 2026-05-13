import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.dataset import CompanyPageDataset, custom_collate_fn
from src.model import HierarchicalAttentionBERT

# --- パスの設定 ---
INPUT_CSV = os.path.join(BASE_DIR, "data", "splitted_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "label_mappings.json")
OUTPUT_IMG = os.path.join(BASE_DIR, "data", "confusion_matrix.png")

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 4 # 推論時はメモリ消費が少ないので4でも動く可能性が高いです（厳しければ1にしてください）

def main():
    print("--- 🔍 AIモデルのテスト（評価）を開始します ---")

    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 2. マッピング辞書の読み込み
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    revenue2id = mappings["revenue2id"]
    category2id = mappings["category2id"]
    # IDからラベル（クラス名）に戻すための辞書
    id2revenue = {int(v): k for k, v in revenue2id.items()}

    num_classes = len(revenue2id)
    num_categories = len(category2id)

    # 3. データの準備（学習時と全く同じ検証用データを再現する）
    df = pd.read_csv(INPUT_CSV)
    df['label'] = df['revenue_class'].astype(str).map(revenue2id)
    df['category_id'] = df['page_category'].astype(str).map(category2id)

    company_labels_df = df.drop_duplicates(subset=['company_name'])[['company_name', 'label']]
    
    # ※ train.pyで test_size=0.1 に変更した場合は、ここも 0.1 に合わせてください！
    _, val_companies = train_test_split(
        company_labels_df['company_name'], 
        test_size=0.2, 
        random_state=42,
        stratify=company_labels_df['label']
    )
    
    val_df = df[df['company_name'].isin(val_companies)]
    print(f"評価対象（検証用データ）: {len(val_companies)}社")

    # 4. モデルとデータローダーの準備
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_dataset = CompanyPageDataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    model = HierarchicalAttentionBERT(model_name=MODEL_NAME, num_categories=num_categories, num_classes=num_classes)
    
    # 🌟 保存した重み（脳みそ）をロード
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 5. 推論の実行（アンサンブル）
    company_predictions = {}
    company_labels = {}

    print("推論を実行中...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            category_ids = batch['category_ids'].to(device)
            labels = batch['labels']
            names = batch['company_names']

            logits = model(input_ids, attention_mask, chunk_mask, category_ids)
            probs = torch.softmax(logits, dim=1)

            for i, name in enumerate(names):
                if name not in company_predictions:
                    company_predictions[name] = []
                    company_labels[name] = labels[i].item()
                company_predictions[name].append(probs[i].cpu().numpy())

    # 企業ごとの最終予測を決定
    y_true = []
    y_pred = []
    
    for name, pred_list in company_predictions.items():
        avg_prob = np.mean(pred_list, axis=0)
        final_pred = np.argmax(avg_prob)
        y_true.append(company_labels[name])
        y_pred.append(final_pred)

    # 6. 結果の表示と保存
    target_names = [id2revenue[i] for i in range(num_classes)]
    
    print("\n" + "="*50)
    print("📊 評価レポート (Classification Report)")
    print("="*50)
    # 適合率(Precision)、再現率(Recall)、F値(F1-score)を表示
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 混同行列（Confusion Matrix）の画像生成
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Class (実際の収益)')
    plt.xlabel('Predicted Class (AIの予測)')
    plt.title('Confusion Matrix (混同行列)')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    
    print(f"\n✨ 混同行列の画像を保存しました: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()