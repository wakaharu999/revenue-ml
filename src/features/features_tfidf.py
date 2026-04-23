import pandas as pd
import json
import os
import fugashi
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 設定と定数
# ==========================================
JSONL_PATH = './data/crawled_data.jsonl'
OUTPUT_TFIDF_PATH = './data/tfidf_features.parquet' 
TFIDF_MODEL_PATH = './data/models/tfidf_vectorizer.joblib'

TFIDF_MAX_FEATURES = 500

# ==========================================
# トークナイザ（名詞のみ・セグフォ対策・Pickle対応）
# ==========================================
class NounTokenizer:
    def __init__(self):
        self.tagger = fugashi.Tagger() # type: ignore

    def __call__(self, text):
        words = []
        chunk_size = 5000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            for word in self.tagger(chunk):
                if hasattr(word.feature, 'pos1'):
                    pos = word.feature.pos1
                else:
                    pos = word.feature.split(',')[0]
                if pos == '名詞':
                    words.append(word.surface)
        return words
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'tagger' in state:
            del state['tagger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tagger = fugashi.Tagger() # type: ignore
    
def main():
    print("1. クロールデータ（JSONL）を読み込み中...")
    crawled_data = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                crawled_data.append(json.loads(line))
    
    df_raw = pd.DataFrame(crawled_data)
    
    print("2. 企業ごとにテキストの結合とメタデータの抽出を実行中...")
    # 企業名でグルーピング
    grouped = df_raw.groupby('company_name')
    
    company_texts = []
    company_metadata = [] # 🌟 ここに {company_name, url, revenue_class} を貯める
    
    for company, group in tqdm(grouped):
        # テキストの結合
        all_text = " ".join(group['text_content'].fillna("").tolist())
        company_texts.append(all_text)
        
        # 🌟 メタデータの取得（最初の1行からURLとクラスを代表値として取得）
        meta = {
            'company_name': company,
            'url': group['url'].iloc[0],
            'revenue_class': group['revenue_class'].iloc[0]
        }
        company_metadata.append(meta)

    print("3. TF-IDFの学習（辞書作成）と変換を実行中...")
    vectorizer = TfidfVectorizer(
        tokenizer=NounTokenizer(),
        max_features=TFIDF_MAX_FEATURES,
        max_df=0.90,
        min_df=3,
        token_pattern=None # type: ignore
    )
    
    tfidf_matrix = vectorizer.fit_transform(company_texts).toarray() # type: ignore
    
    print("4. 結果の統合とParquetへの書き出し...")
    # モデルの保存
    os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
    joblib.dump(vectorizer, TFIDF_MODEL_PATH)
    
    # 🌟 メタデータとTF-IDFベクトルを1つのリストにまとめる
    final_rows = []
    for i, meta in enumerate(company_metadata):
        # 既存のメタデータ（name, url, class）をコピー
        row = meta.copy()
        # TF-IDFベクトル（500次元）を追加
        for j, val in enumerate(tfidf_matrix[i]):
            row[f'tfidf_vec_{j}'] = val
        final_rows.append(row)
        
    df_final = pd.DataFrame(final_rows)
    
    # 保存
    df_final.to_parquet(OUTPUT_TFIDF_PATH, index=False)
    
    print(f"✅ 処理が完了しました！")
    print(f"データ形状: {df_final.shape}")
    print(f"先頭のカラム: {df_final.columns[:5].tolist()}")
    print(f"保存先: {OUTPUT_TFIDF_PATH}")

if __name__ == "__main__":
    main()