"""
本番用モデル訓練スクリプト
Model 2 (Text + Structural Features Late Fusion) を全データで学習し、
SavedModel 形式でモデルと関連ファイルを保存
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# 1. 設定
# ============================================================
RANDOM_SEED = 333
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/revenue_model')
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'tf_model')
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# 乱数シード設定
tf.keras.utils.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 2. データの読み込みと前処理
# ============================================================
def load_and_prepare_data():
    """データを読み込み、テキストベクトル・構造特徴量を抽出"""
    print("📂 Loading data...")
    
    # Parquet ファイルを読み込み
    df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))
    df_tfidf = pd.read_parquet(os.path.join(DATA_DIR, 'tfidf_features.parquet'))
    
    # 不要な列を削除
    cols_to_drop = [c for c in ['revenue_class', 'url', 'page_category'] 
                    if c in df_tfidf.columns]
    df_tfidf_clean = df_tfidf.drop(columns=cols_to_drop)
    
    # 企業名でインナージョイン
    df_merged = pd.merge(df, df_tfidf_clean, on='company_name', how='inner')
    
    print(f"✓ Merged data shape: {df_merged.shape}")
    
    # テキストベクトルの抽出（HuggingFace の _vec_ カラム）
    hf_cols = [c for c in df_merged.columns 
               if '_vec_' in c and not c.startswith('tfidf_vec_')]
    X_text = df_merged[hf_cols].fillna(0).to_numpy(dtype=np.float32)
    
    # 構造特徴量の抽出
    struct_cols = [
        c for c in df_merged.columns 
        if not c.startswith('tfidf_vec_') 
        and '_vec_' not in c 
        and c not in ['company_name', 'revenue_class', 'url', 'page_category']
        and pd.api.types.is_numeric_dtype(df_merged[c])
    ]
    X_struct = df_merged[struct_cols].fillna(0).to_numpy(dtype=np.float32)
    
    # ラベルエンコーディング
    le = LabelEncoder()
    y = le.fit_transform(df_merged['revenue_class'])
    y = np.array(y, dtype=np.int32)
    
    num_classes = len(le.classes_)
    
    print(f"✓ Text vectors shape       : {X_text.shape}")
    print(f"✓ Structural features shape: {X_struct.shape}")
    print(f"✓ Number of classes       : {num_classes}")
    print(f"✓ Classes                 : {le.classes_}")
    print(f"✓ Text vector columns     : {len(hf_cols)}")
    print(f"✓ Structural columns      : {len(struct_cols)}")
    
    return X_text, X_struct, y, le, struct_cols, hf_cols, num_classes


# ============================================================
# 3. Model 2: Two-Tower Late Fusion Architecture
# ============================================================
def build_model_2(text_dim, struct_dim, num_classes):
    """
    Model 2: テキストと構造特徴量を独立して処理してから結合する融合モデル
    
    Args:
        text_dim: テキストベクトルの次元数
        struct_dim: 構造特徴量の次元数
        num_classes: クラス数
        
    Returns:
        keras.Model: コンパイル済みモデル
    """
    # --- 入力層 ---
    text_input = keras.Input(shape=(text_dim,), name="text_vectors")
    struct_input = keras.Input(shape=(struct_dim,), name="structural_features")
    
    # --- テキスト処理用MLP ---
    x_text = layers.Dense(256, activation="relu")(text_input)
    x_text = layers.BatchNormalization()(x_text)
    x_text = layers.Dropout(0.3)(x_text)
    
    x_text = layers.Dense(64, activation="relu")(x_text)
    x_text = layers.BatchNormalization()(x_text)
    x_text = layers.Dropout(0.3)(x_text)
    
    # --- 構造特徴量処理用MLP ---
    x_struct = layers.Dense(64, activation="relu")(struct_input)
    x_struct = layers.BatchNormalization()(x_struct)
    x_struct = layers.Dropout(0.2)(x_struct)
    
    # --- 結合 (Late Fusion) ---
    combined = layers.Concatenate(name="late_fusion")([x_text, x_struct])
    
    # --- 結合後のMLP ---
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    
    outputs = layers.Dense(num_classes, activation="softmax")(z)
    
    # モデルのコンパイル
    model = keras.Model(inputs=[text_input, struct_input], outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ============================================================
# 4. モデルの訓練
# ============================================================
def train_model(X_text, X_struct, y, num_classes):
    """
    モデル2を全データで訓練
    
    Args:
        X_text: テキストベクトル配列
        X_struct: 構造特徴量配列
        y: ラベル配列
        num_classes: クラス数
        
    Returns:
        model: 訓練済みモデル
        scaler: 構造特徴量の正規化器
    """
    print("\n🚀 Training Model 2 (Late Fusion: Text + Structural Features)...")
    
    # 構造特徴量の正規化（情報漏洩防止のため全データで fittingしない）
    # 本番運用では、全データで fit_transform するのが標準
    scaler = StandardScaler()
    X_struct_scaled = scaler.fit_transform(X_struct)
    
    # モデルのビルド
    model = build_model_2(X_text.shape[1], X_struct.shape[1], num_classes)
    
    # 早期停止の設定
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # 訓練
    history = model.fit(
        {"text_vectors": X_text, "structural_features": X_struct_scaled},
        y,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("✓ Training completed")
    
    return model, scaler


# ============================================================
# 5. モデルと関連ファイルの保存
# ============================================================
def save_model_artifacts(model, scaler, le, struct_cols, hf_cols, num_classes):
    """
    モデルと関連ファイルを保存
    
    Args:
        model: 訓練済みモデル
        scaler: 構造特徴量の正規化器
        le: ラベルエンコーダー
        struct_cols: 構造特徴量のカラム名リスト
        hf_cols: テキストベクトルのカラム名リスト
        num_classes: クラス数
    """
    print("\n💾 Saving model artifacts...")
    
    # モデルディレクトリの作成
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. TensorFlow モデルを SavedModel 形式で保存
    print(f"  → Saving SavedModel to {SAVED_MODEL_DIR}")
    model.save(SAVED_MODEL_DIR, save_format='tf')
    
    # 2. LabelEncoder を保存
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"  ✓ Label encoder saved: {le_path}")
    
    # 3. StandardScaler を保存
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # 4. メタデータ（設定）を JSON で保存
    metadata = {
        "model_name": "revenue_prediction_model_v2",
        "model_type": "two_tower_late_fusion",
        "description": "Text + Structural Features Late Fusion Model for Revenue Classification",
        "num_classes": num_classes,
        "classes": le.classes_.tolist(),
        "text_vector_dim": len(hf_cols),
        "structural_features_dim": len(struct_cols),
        "structural_feature_columns": struct_cols,
        "text_vector_columns": hf_cols,
        "random_seed": RANDOM_SEED,
        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "validation_split": VALIDATION_SPLIT
        },
        "architecture": {
            "text_tower": {
                "layers": [
                    {"type": "Dense", "units": 256, "activation": "relu"},
                    {"type": "BatchNormalization"},
                    {"type": "Dropout", "rate": 0.3},
                    {"type": "Dense", "units": 64, "activation": "relu"},
                    {"type": "BatchNormalization"},
                    {"type": "Dropout", "rate": 0.3}
                ]
            },
            "struct_tower": {
                "layers": [
                    {"type": "Dense", "units": 64, "activation": "relu"},
                    {"type": "BatchNormalization"},
                    {"type": "Dropout", "rate": 0.2}
                ]
            },
            "fusion": {
                "method": "late_fusion_concatenate"
            },
            "output_tower": {
                "layers": [
                    {"type": "Dense", "units": 256, "activation": "relu"},
                    {"type": "BatchNormalization"},
                    {"type": "Dropout", "rate": 0.4},
                    {"type": "Dense", "units": num_classes, "activation": "softmax"}
                ]
            }
        }
    }
    
    metadata_path = os.path.join(MODEL_DIR, 'model_config.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Model config saved: {metadata_path}")
    
    print("\n✅ All artifacts saved successfully!")
    print(f"   Model directory: {MODEL_DIR}")
    print(f"   SavedModel path: {SAVED_MODEL_DIR}")



# ============================================================
# 7. メイン処理
# ============================================================
def main():
    """メイン処理"""
    print("="*60)
    print("🤖 Model Training Pipeline - Model 2 (Late Fusion)")
    print("="*60)
    
    # データの読み込み
    X_text, X_struct, y, le, struct_cols, hf_cols, num_classes = load_and_prepare_data()
    
    # モデルの訓練
    model, scaler = train_model(X_text, X_struct, y, num_classes)
    
    # モデルと関連ファイルの保存
    save_model_artifacts(model, scaler, le, struct_cols, hf_cols, num_classes)
    
    print("\n" + "="*60)
    print("✨ Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
