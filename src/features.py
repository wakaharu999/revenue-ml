import pandas as pd 
import numpy as np #
import json
import re
import os
from tqdm import tqdm # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

# ==========================================
# 設定と定数
# ==========================================
JSONL_PATH = '/app/data/crawled_data.jsonl'
CSV_PATH = '/app/data/train.csv'
OUTPUT_PATH = '/app/data/train_features.parquet' # 列数が多いためParquet形式で保存

CATEGORIES = ['top', 'about', 'history', 'business', 'ir', 'recruit', 'news']
CHUNK_SIZE = 400

# ==========================================
# 1. 追加特徴量（キーワード・メタデータ）抽出関数
# ==========================================
def extract_meta_features(df_crawled):
    print("--- 1. 追加特徴量（メタデータ＆キーワード＆数値抽出）の抽出 ---")
    
    feature_rows = []
    grouped = df_crawled.groupby('company_name')
    
    # 正規表現で最初に見つかった数値を安全に抽出するヘルパー関数
    def get_first_num(pattern, text, default=0):
        match = re.search(pattern, text)
        # match が存在し、かつ 1番目のカッコの中身が None でないことを確認
        if match and match.group(1) is not None:
            num_str = re.sub(r'[^\d.]', '', match.group(1))
            try:
                return float(num_str)
            except ValueError:
                return default
        return default

    for company, group in tqdm(grouped):
        # 企業内の全テキストを結合（ページごとのテキストも保持）
        text_dict = {cat: " ".join(group[group['page_category'] == cat]['text_content'].fillna("").tolist()) for cat in CATEGORIES}
        all_text = " ".join(text_dict.values())
        total_length = len(all_text) + 1 
        
        features = {'company_name': company}
        
        # --------------------------------------------------
        # 0. 基本の構造特徴（文字数とページの有無）
        # --------------------------------------------------
        for cat in CATEGORIES:
            cat_text = text_dict[cat]
            features[f'has_{cat}'] = 1 if len(cat_text) > 0 else 0
            features[f'{cat}_length_log'] = np.log1p(len(cat_text))
            features[f'{cat}_ratio'] = len(cat_text) / total_length

        # --------------------------------------------------
        # 1. グローバル展開力
        # --------------------------------------------------
        global_words = r'グローバル|global|海外|世界|多国籍|現地|国際|輸出|輸入|北米|アメリカ|欧州|ヨーロッパ|中国|ASEAN|アジア|中東|アフリカ|オセアニア'
        features['global_word_score'] = np.log1p(len(re.findall(global_words, all_text)))
        features['overseas_sales_ratio'] = get_first_num(r'海外売上(?:高|比率).*?([0-9\.]+)%', all_text)
        features['global_bases'] = get_first_num(r'(?:海外|世界)(?:拠点)?[\s\w]{0,10}?([0-9,]+)(?:ヶ所|箇所|拠点)', all_text)

        # --------------------------------------------------
        # 2. 歴史・伝統
        # --------------------------------------------------
        era_words = r'明治|大正|昭和|平成|令和'
        features['era_word_score'] = np.log1p(len(re.findall(era_words, all_text)))
        features['founding_year'] = get_first_num(r'(?:創業|創立|設立|発祥|始業).*?((?:18|19|20)[0-9]{2})年', all_text, default=np.nan) # type: ignore
        # --------------------------------------------------
        # 3. M&A・提携・研究開発
        # --------------------------------------------------
        ma_words = r'M&A|買収|合併|統合|提携|アライアンス|alliance|パートナー|partner|協業|研究|産学|R&D|共同研究|イノベーション|innovation|特許|知的財産|大学|TOB|子会社|グループ会社|プライム'
        features['ma_word_score'] = np.log1p(len(re.findall(ma_words, all_text)))
        features['subsidiaries_count'] = get_first_num(r'(?:連結子会社|グループ会社|関係会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text)
        features['partners_count'] = get_first_num(r'(?:提携先|パートナー数|協力会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text)
        # --------------------------------------------------
        # 4. 市場シェア・優位性
        # --------------------------------------------------
        share_words = r'最大手|トップ|top|No\.1|ナンバーワン|リード|リーディングカンパニー|唯一|独自|オンリー|特許|only|best|首位'
        features['market_leader_score'] = np.log1p(len(re.findall(share_words, all_text)))
        features['domestic_share_pct'] = get_first_num(r'(?:国内|世界|市場)シェア.*?([0-9\.]+)%', all_text)

        # --------------------------------------------------
        # 5. 事業規模（事業数・金額規模）
        # --------------------------------------------------
        # 「〇〇事業」という単語のユニーク数をカウント（事業の多角化指標）
        business_types = set(re.findall(r'([一-龥]{2,6}事業)', all_text))
        features['num_business_types'] = len(business_types)
        
        features['money_cho_score'] = np.log1p(len(re.findall(r'兆円|数兆|兆', all_text)))
        features['money_oku_score'] = np.log1p(len(re.findall(r'億円|数百億|億', all_text)))
        
        # --------------------------------------------------
        # 6. 人事系（規模と福利厚生）
        # --------------------------------------------------
        welfare_words = r'福利厚生|ワークライフバランス|くるみん|ホワイト|テレワーク|フレックス|ダイバーシティ|女性|育児|介護|休暇|健康|QOL|健康|DX'
        features['welfare_word_score'] = np.log1p(len(re.findall(welfare_words, all_text)))
        features['employees_count'] = get_first_num(r'(?:連結|グループ|正社員|就業|全社)?(?:従業員|人員|社員|スタッフ)(?:数|合計)?[\s\w:：]{0,10}?([0-9,]+)(?:名|人)', all_text)
        job_types = set(re.findall(r'([一-龥]{2,5}職)', all_text))
        features['num_job_types'] = len(job_types)

        # --------------------------------------------------
        # 7. 社会責任・ガバナンス
        # --------------------------------------------------
        gov_words = r'コーポレートガバナンス|コンプライアンス|内部統制|リスクマネジメント|監査|サステナビリティ|SDGs|ESG|CSR|環境保全|社会貢献|カーボンニュートラル|脱炭素'
        features['governance_score'] = np.log1p(len(re.findall(gov_words, all_text)))
        
        feature_rows.append(features)
        
    return pd.DataFrame(feature_rows)

# ==========================================
# 2. Hugging Face Transformersによるテキストベクトル化関数
# ==========================================
def extract_category_vectors(df_crawled):
    print("--- 2. テキストのベクトル化 (multilingual-e5-base) ---")
    #multilingual-e5-baseを採用
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    
    # 企業×カテゴリ ごとのチャンクリストを作成
    # 辞書構造: { '企業名': { 'about': ['passage: チャンク1', ...], 'ir': [...] } }
    chunk_dict = {}
    for _, row in df_crawled.iterrows():
        comp = row['company_name']
        cat = row['page_category']
        text = str(row['text_content'])
        
        if comp not in chunk_dict:
            chunk_dict[comp] = {c: [] for c in CATEGORIES}
            
        # 400文字ずつに分割し、"passage: " を付与
        chunks = [f"passage: {text[i:i+CHUNK_SIZE]}" for i in range(0, len(text), CHUNK_SIZE)]
        chunk_dict[comp][cat].extend(chunks)

    # ベクトル化してMean Pooling
    vector_rows = []
    for comp, cat_chunks in tqdm(chunk_dict.items()):
        comp_vecs = {'company_name': comp}
        
        for cat in CATEGORIES:
            chunks = cat_chunks[cat]
            if len(chunks) == 0:
                # ページが存在しない場合は 768次元のゼロベクトル
                pooled_vec = np.zeros(768)
            else:
                # チャンクを一括でベクトル化（バッチ処理）
                embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
                # Mean Pooling (チャンク群の平均をとる)
                pooled_vec = np.mean(embeddings, axis=0)
                
            # 列名をつけて保存 (例: about_vec_0, about_vec_1 ... about_vec_767)
            for i, val in enumerate(pooled_vec):
                comp_vecs[f'{cat}_vec_{i}'] = val
                
        vector_rows.append(comp_vecs)
        
    return pd.DataFrame(vector_rows)

def main():
    df_train = pd.read_csv(CSV_PATH)
    
    # すでに保存済みのデータを読み込む（再開用）
    if os.path.exists(OUTPUT_PATH):
        df_already = pd.read_parquet(OUTPUT_PATH)
        done_companies = df_already['company_name'].unique().tolist()
    else:
        df_already = pd.DataFrame()
        done_companies = []

    # 未処理の企業だけを抽出
    df_todo = df_train[~df_train['company_name'].isin(done_companies)]
    if len(df_todo) == 0:
        print("すべての企業の処理が完了しています！")
        return

    # JSONLの読み込み
    crawled_data = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data['company_name'] in df_todo['company_name'].values:
                    crawled_data.append(data)
    df_crawled_all = pd.DataFrame(crawled_data)

    # 特徴量抽出の実行
    BATCH_SIZE = 10  # 10社ごとに保存する
    for i in range(0, len(df_todo), BATCH_SIZE):
        batch_target = df_todo.iloc[i : i + BATCH_SIZE]
        target_names = batch_target['company_name'].tolist()
        print(f"\nバッチ実行中: {i+1}〜{i+len(target_names)} 社目 / 残り {len(df_todo)} 社")

        df_crawled_batch = df_crawled_all[df_crawled_all['company_name'].isin(target_names)]
        
        # 特徴量抽出
        df_meta_batch = extract_meta_features(df_crawled_batch)
        df_vecs_batch = extract_category_vectors(df_crawled_batch)
        
        # 結合
        df_final_batch = pd.merge(batch_target, df_meta_batch, on='company_name', how='left')
        df_final_batch = pd.merge(df_final_batch, df_vecs_batch, on='company_name', how='left')
        
        # 過去のデータと結合して保存
        df_already = pd.concat([df_already, df_final_batch], ignore_index=True)
        df_already.to_parquet(OUTPUT_PATH, index=False)
        print(f"✅ チェックポイント保存完了: {OUTPUT_PATH}")

    print(f"✨ すべての処理が完了しました！ データ形状: {df_already.shape}")

if __name__ == "__main__":
    main()