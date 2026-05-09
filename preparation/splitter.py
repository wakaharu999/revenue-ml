import os
import json
import pandas as pd
import csv
# --- パスの設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSONL = os.path.join(BASE_DIR, "data", "crawled_data.jsonl")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "splitted_dataset.csv")

# BERTが一度に読める上限に安全に収めるため、400文字ずつに分割
CHUNK_SIZE = 400 

def main():
    print("--- 🔪 テキストのチャンキング（分割）を開始します ---")
    
    if not os.path.exists(INPUT_JSONL):
        print(f"【エラー】{INPUT_JSONL} が見つかりません。")
        return

    processed_data = []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            company = record.get("company_name", "")
            rev_class = record.get("revenue_class", "")
            category = record.get("page_category", "")
            text = record.get("text_content", "")

            # エラー記録や、空のテキストは学習に使えないのでスキップ
            if category == "timeout_error" or not text:
                continue

            # テキストをCHUNK_SIZE（400文字）ごとにスライスしていく
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i : i + CHUNK_SIZE]
                
                # 短すぎるゴミデータ（50文字未満）は、文脈がなくノイズになるので捨てる
                if len(chunk) < 50:
                    continue

                processed_data.append({
                    "company_name": company,
                    "revenue_class": rev_class,  # AIに予測させたい正解ラベル（S, A, Bなど）
                    "page_category": category,
                    "text": chunk                # AIに読ませるテキスト
                })

    # DataFrameに変換してCSVとして保存
    print(f"データ変換中... (全 {len(processed_data)} 件のチャンクを生成)")
    df = pd.DataFrame(processed_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", escapechar="\\", quoting=csv.QUOTE_ALL)
    print(f" データセット作成完了！")
    print(f"総チャンク数（学習データ数）: {len(df)} 件")
    print(f"保存先: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()