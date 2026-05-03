# step1: EDINETDB APIから企業の売上ランキングデータを500件取得し、JSONファイルに保存するスクリプト
# step2: 保存したJSONファイルから企業URLと売上ランクを抽出し、CSVファイルに保存するスクリプト

import os
import json
import csv
import time
import requests 
from dotenv import load_dotenv 
from urllib.parse import urlparse
load_dotenv()
API_KEY = os.getenv("EDB_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# 保存先
JSON_PATH = "../data/raw_api_data.json"
CSV_PATH = "../data/company.csv"

# ==========================================
# STEP 1: JSONファイルの作成
# ==========================================
def step1_fetch_and_save_json():
    if not API_KEY:
        print("【エラー】EDINET_API_KEYが設定されていません。")
        return False

    API_URL = "https://edinetdb.jp/v1/rankings/revenue?limit=1500"
    headers = {"X-API-KEY": API_KEY}
    
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code != 200:
            print(f"【エラー】API取得失敗: ステータスコード {response.status_code}")
            print(f"詳細: {response.text}")
            return False

        data = response.json()

        # データが辞書型かリスト型かに対応
        items = data.get("data", []) if isinstance(data, dict) else data
        extracted_data = []

        # APIのレスポンスから「企業名」と「売上」を抽出
        for item in items:
            name = item.get("name")
            revenue = item.get("value")  
            extracted_data.append({"name": name, "revenue": revenue})

        # JSONファイルとして保存
        os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSONを保存しました: {JSON_PATH} ({len(extracted_data)}件)")
        return True

    except Exception as e:
        print(f"【通信エラー】APIへの接続中にエラーが発生しました: {e}")
        return False

def search_official_url_serper(name):
    query = f"{name} 企業情報" 
    url = "https://google.serper.dev/search"
    
    # SerperAPIに送る設定
    payload = json.dumps({
      "q": query,
      "gl": "jp",
      "hl": "ja",
      "num": 5  # 上位5件を取得
    })
    headers = {
      'X-API-KEY': SERPER_API_KEY,
      'Content-Type': 'application/json'
    }

    # リクルートサイト等を弾く用のリスト
    blacklist = [
        "wikipedia.org", "rikunabi.com", "mynavi.jp", "doda.jp", "en-japan.com", 
        "type.jp", "prtimes.jp", "nikkei.com", "toyokeizai.net", "yahoo.co.jp", 
        "instagram.com", "twitter.com", "x.com", "facebook.com", "tiktok.com", 
        "youtube.com", "amazon.co.jp", "bing.com", "irbank.net", "baseconnect.in",
        "macloud.jp", "kabu.com", "shukatsu", "houjin.jp", "syukatsu-kaigi.jp"
    ]
    
    try:
        # POSTリクエストでAPIを叩く
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code != 200:
            print(f" [APIエラー: {response.status_code}] ", end="")
            return "API_ERROR"
            
        data = response.json()
        items = data.get("organic", [])
        
        valid_urls = []
        for item in items:
            link = item.get("link", "")
            
            # PDFや採用ページを除外
            if link.lower().endswith(".pdf"):
                continue
            if "career" in link.lower() or "recruit" in link.lower():
                continue
                
            try:
                domain = urlparse(link).netloc.lower()
            except:
                continue
                
            is_bad_domain = any(bad in domain for bad in blacklist)
            
            if not is_bad_domain:
                valid_urls.append(link)
                
        if valid_urls:
            return min(valid_urls, key=len)
            
    except Exception as e:
        print(f" [通信エラー: {e}] ", end="")
        
    return "NOT_FOUND"

# ==========================================
# STEP 2: JSONからCSVを作成
# ==========================================
def step2_generate_csv():
    print("--- [STEP 2] 企業名からURLを検索してCSVを作成します ---")

    if not SERPER_API_KEY:
        print("【エラー】SerperのAPIキーが設定されていません。")
        return

    if not os.path.exists(JSON_PATH):
        print("【エラー】JSONファイルが見つかりません。")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        companies = json.load(f)

    
    print(f"テストモード: 全 {len(companies)} 件中、最初の {len(companies)} 件だけを処理します...\n")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["company_name", "url", "revenue_class"])

        success_count = 0
        
        for comp in companies:
            name = comp.get("name")
            revenue = comp.get("revenue")

            if not name or not revenue:
                continue

            if revenue >= 10000000:
                rev_class = "S"
            elif revenue >= 1000000:
                rev_class = "A"
            elif revenue >= 500000:
                rev_class = "B"
            else:
                rev_class = "C"

            print(f"検索中: {name} ... ", end="", flush=True)

            url = search_official_url_serper(name)
            
            if url == "API_ERROR":
                print("\n【警告】APIでエラーが発生しました")
                break
                
            if url != "NOT_FOUND":
                print("OK")
                writer.writerow([name, url, rev_class])
                success_count += 1
            else:
                print("見つかりませんでした")
                writer.writerow([name, "NOT_FOUND", rev_class])
                
            time.sleep(0.5)

    print(f"\nテスト処理が完了しました。 {success_count} 件のデータを保存しました。")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    step1_fetch_and_save_json()
    step2_generate_csv()