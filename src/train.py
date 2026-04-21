import os
import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from transformers import AutoTokenizer # type: ignore

# 乱数シードの固定（再現性確保）
tf.keras.utils.set_random_seed(42)

# ==========================================
# 0. 特徴量の簡単選択設定（ここでON/OFFを管理）
# ==========================================
FEATURE_CONFIG = {
    "text_pages": {
        "top": True,
        "about": True,
        "history": True,
        "business": True,
        "ir": True,
        "recruit": True,
        "news": True
    },
    "structural_categories": {
        "global_power": True,
        "history_brand": True,
        "ma_alliance": True,
        "market_share": True,
        "business_scale": True,
        "hr_welfare": True,
        "governance": True
    }
}

STRUCT_COL_MAP = {
    "global_power": ['global_word_score', 'overseas_sales_ratio', 'global_bases'],
    "history_brand": ['era_word_score', 'founding_year'],
    "ma_alliance": ['ma_word_score', 'subsidiaries_count', 'partners_count'],
    "market_share": ['market_leader_score', 'domestic_share_pct'],
    "business_scale": ['num_business_types', 'money_cho_score', 'money_oku_score'],
    "hr_welfare": ['welfare_word_score', 'employees_count', 'num_job_types'],
    "governance": ['governance_score']
}

# ==========================================
# 1. データの読み込みと特徴量抽出
# ==========================================
print("データを読み込み、設定に基づいて特徴量を抽出します...")
df = pd.read_parquet('../data/train_features.parquet')

# ラベルの数値化
le = LabelEncoder()
y = le.fit_transform(df['revenue_class'])
num_classes = len(le.classes_) 

selected_text_cols = []
for page, is_active in FEATURE_CONFIG["text_pages"].items():
    if is_active:
        vec_cols = [c for c in df.columns if c.startswith(f"{page}_vec_")]
        selected_text_cols.extend(vec_cols)
        selected_text_cols.extend([c for c in df.columns if c in [f"has_{page}", f"{page}_length_log", f"{page}_ratio"]])

selected_struct_cols = []
for category, is_active in FEATURE_CONFIG["structural_categories"].items():
    if is_active:
        cols = [c for c in STRUCT_COL_MAP[category] if c in df.columns]
        selected_struct_cols.extend(cols)

X_text = df[selected_text_cols].fillna(0).values
X_struct = df[selected_struct_cols].fillna(0).values

print(f"テキスト特徴量 {X_text.shape[1]}次元, 構造的特徴量 {X_struct.shape[1]}次元")

# ==========================================
# 1.5 データの分割とスケーリング（★一番重要だった抜け落ち部分）
# ==========================================
print("学習データと検証データに分割しています...")
X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    X_text, X_struct, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_struct_train_scaled = scaler.fit_transform(X_struct_train)
X_struct_val_scaled = scaler.transform(X_struct_val)


# ==========================================
# 2. モデル構築（マルチ入力アーキテクチャ）
# ==========================================
print("モデルを構築しています...")

text_input = keras.Input(shape=(X_text.shape[1],), name="text_vectors")
x1 = layers.Dense(256, activation="relu")(text_input)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.4)(x1)
x1 = layers.Dense(128, activation="relu")(x1)

struct_input = keras.Input(shape=(X_struct.shape[1],), name="structural_features")
x2 = layers.Dense(64, activation="relu")(struct_input)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.3)(x2)

combined = layers.Concatenate()([x1, x2])

z = layers.Dense(128, activation="relu")(combined)
z = layers.Dropout(0.3)(z)
z = layers.Dense(64, activation="relu")(z)
output = layers.Dense(num_classes, activation="softmax", name="prediction")(z)

model = keras.Model(inputs=[text_input, struct_input], outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 3. 学習（Training）
# ==========================================
print("学習を開始します...")
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = model.fit(
    {"text_vectors": X_text_train, "structural_features": X_struct_train_scaled},
    y_train,
    validation_data=(
        {"text_vectors": X_text_val, "structural_features": X_struct_val_scaled},
        y_val
    ), # type: ignore
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 4. 評価（Evaluation）
# ==========================================
print("\nモデルの評価を実行します...")
y_pred_prob = model.predict({"text_vectors": X_text_val, "structural_features": X_struct_val_scaled}) # type: ignore
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\n【クラス別詳細レポート】")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 混同行列の描画
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.title('Confusion Matrix (Multi-Input DNN)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ==========================================
# 5. モデルとトークナイザーの保存
# ==========================================
print("\nモデル一式を保存しています...")
os.makedirs('../models/revenue_model', exist_ok=True)

model.save('../models/revenue_model/tf_model.keras')
print("・TFモデルを保存しました (.keras)")

joblib.dump(scaler, '../models/revenue_model/scaler.pkl')
joblib.dump(le, '../models/revenue_model/label_encoder.pkl')
print("・スケーラーとラベルエンコーダーを保存")

MODEL_NAME = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained('../models/revenue_model/tokenizer')
print("・トークナイザー設定一式を保存")

print(" すべてのプロセスが完了")