import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from src.dataset import CompanyPageDataset, custom_collate_fn
from src.model import HierarchicalAttentionBERT

# --- パスの設定 ---
INPUT_CSV = os.path.join(BASE_DIR, "data", "splitted_dataset.csv") # ※ファイル名が正しいか確認してください
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "data", "saved_model")

# --- ハイパーパラメータ ---
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BATCH_SIZE = 1 # メモリが溢れてしまったので少なく
ACCUMULATION_STEPS = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 🌟 新規追加：企業単位のアンサンブル評価関数
def validate_ensemble(model, val_loader, device, criterion):
    model.eval()
    company_predictions = {} 
    company_labels = {}      
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            category_ids = batch['category_ids'].to(device)
            labels = batch['labels'].to(device)
            names = batch['company_names'] # dataset.pyで追加した名前情報

            logits = model(input_ids, attention_mask, chunk_mask, category_ids)
            
            # Lossの計算
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)

            # 企業ごとに確率をストックする
            for i, name in enumerate(names):
                if name not in company_predictions:
                    company_predictions[name] = []
                    company_labels[name] = labels[i].item()
                company_predictions[name].append(probs[i].cpu().numpy())

    # 企業ごとの平均アンサンブル
    correct = 0
    for name, pred_list in company_predictions.items():
        avg_prob = np.mean(pred_list, axis=0) # ページごとの確率を平均
        final_pred = np.argmax(avg_prob)      # 最も高いクラスを最終予測とする
        if final_pred == company_labels[name]:
            correct += 1
            
    avg_val_loss = total_val_loss / len(val_loader)
    ensemble_acc = correct / len(company_predictions)
    
    return avg_val_loss, ensemble_acc

        # 順伝播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)

def main():
    print("---  AIの学習（Training）を開始します ---")

    # 1. デバイスの設定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Apple Silicon GPU (MPS) を使用します")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🟢 NVIDIA GPU (CUDA) を使用します")
    else:
        device = torch.device("cpu")
        print("⚪ CPU を使用します。")

    # 2. データの準備と分割
    print("データを読み込み中...")
    df = pd.read_csv(INPUT_CSV)
    
    import json
    # ラベルエンコーディング：文字列のラベルを数値に変換
    unique_revenues = sorted(df['revenue_class'].astype(str).unique().tolist())
    revenue2id = {label: idx for idx, label in enumerate(unique_revenues)}
    df['label'] = df['revenue_class'].astype(str).map(revenue2id)

    unique_categories = sorted(df['page_category'].astype(str).unique().tolist())
    category2id = {label: idx for idx, label in enumerate(unique_categories)}
    df['category_id'] = df['page_category'].astype(str).map(category2id)

    # 辞書を保存しておきます
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "data", "label_mappings.json"), 'w', encoding='utf-8') as f:
        json.dump({"revenue2id": revenue2id, "category2id": category2id}, f, ensure_ascii=False, indent=2)

    # データリークを防ぐため「企業名」で分割
    unique_companies = df['company_name'].unique()
    train_companies, val_companies = train_test_split(unique_companies, test_size=0.2, random_state=42)
    
    train_df = df[df['company_name'].isin(train_companies)]
    val_df = df[df['company_name'].isin(val_companies)]
    print(f"学習用: {len(train_companies)}社 / 検証用: {len(val_companies)}社")

    # 3. データセットとデータローダーの作成
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = CompanyPageDataset(train_df, tokenizer)
    val_dataset = CompanyPageDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # 4. モデルの準備
    num_classes = len(df['revenue_class'].unique())
    num_categories = len(df['page_category'].unique())
    
    model = HierarchicalAttentionBERT(
        model_name=MODEL_NAME, 
        num_categories=num_categories, 
        num_classes=num_classes
    )
    model.to(device)

    # 5. オプティマイザと損失関数の設定
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    # 6. 学習ループ
    best_val_acc = 0.0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch+1} / {EPOCHS} ========")
        
        # --- 🏋️‍♂️ トレーニングフェーズ ---
        model.train()
        total_loss = 0
        model.zero_grad() # 🌟 修正: エポックの最初と更新直後だけリセットする
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            category_ids = batch['category_ids'].to(device)
            labels = batch['labels'].to(device)

            # 順伝播（予測）
            logits = model(input_ids, attention_mask, chunk_mask, category_ids)
            
            # 誤差計算と逆伝播
            loss = criterion(logits, labels)
            loss = loss / ACCUMULATION_STEPS # 16回分溜めるために割る
            loss.backward()

            total_loss += loss.item() * ACCUMULATION_STEPS

            # パラメータの更新
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad() # 🌟 修正: 溜め込んで更新した後にリセット

            # 🌟 修正: ログ表示時は元のスケールに戻して見やすくする
            if step % 10 == 0 and step > 0:
                print(f"  Batch {step}/{len(train_loader)} - Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"🔹 Average Training Loss: {avg_train_loss:.4f}")

        # --- 🧪 バリデーション（検証）フェーズ ---
        # 🌟 修正: 新しく作ったアンサンブル関数を呼び出す
        val_loss, val_acc = validate_ensemble(model, val_loader, device, criterion)
        
        print(f"🔸 Validation Loss: {val_loss:.4f} | Ensemble Accuracy: {val_acc:.4f}")

        # 最高精度のモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_DIR, "model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✨ 精度が向上しました！モデルを保存: {save_path}")

    print("\n🎉 全ての学習が完了しました！")

if __name__ == "__main__":
    main()