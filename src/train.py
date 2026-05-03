import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.dataset import create_data_loader
from src.model import RevenueClassifier

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """1エポック分の学習を行う関数"""
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in data_loader:
        # データをGPU等のデバイスに転送
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 順伝播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        # 精度と損失を記録
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        # 逆伝播と重みの更新
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader) # type: ignore

def eval_model(model, data_loader, loss_fn, device):
    """検証データでモデルの性能を評価する関数"""
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad(): # 評価時は勾配計算をオフにしてメモリ節約
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader) # type: ignore

def main():
    # --- 設定値 ---
    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    DATA_PATH = "../data/train.csv" 
    SAVE_PATH = "../models/best_model.pth"
    BATCH_SIZE = 16
    MAX_LEN = 512
    EPOCHS = 5
    NUM_CLASSES = 4 # 0〜3の4クラス分類を想定
    # -------------

    # 実行環境に合わせてデバイスを自動選択（MacのM1/M2チップなら mps、NVIDIAなら cuda）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. データの準備
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 訓練データと検証データに 8:2 で分割
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['revenue_class'])

    # トークナイザのロードとデータローダーの作成
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, is_train=True)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE, is_train=False)

    # 2. モデルの初期化
    print("Initializing model...")
    model = RevenueClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
    model = model.to(device)

    # 3. オプティマイザと損失関数の設定
    # ※ BERTのファインチューニングは過学習を防ぐため、非常に小さな学習率(2e-5など)が定石です
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 4. 学習ループ
    print("Starting training...")
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device)
        print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f}")

        # 検証精度が過去最高を記録したら、その時点の重みを保存する
        if val_acc > best_accuracy:
            print(f"★ Validation accuracy improved ({best_accuracy:.4f} -> {val_acc:.4f}). Saving model...")
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            best_accuracy = val_acc

    print("Training complete! Best model saved at:", SAVE_PATH)

if __name__ == "__main__":
    main()