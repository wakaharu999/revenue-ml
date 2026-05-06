# ==========================================
# 企業売上データ＆URL収集スクリプト (gBizINFO API版)
# ==========================================
import os
import json
import csv
import time
import requests
from urllib.parse import quote
from dotenv import load_dotenv

# --- 環境設定 ---
# .env ファイルからAPIキーを読み込む
load_dotenv()

EDB_API_KEY = os.getenv("EDB_API_KEY")
GBIZ_API_KEY = os.getenv("GBIZINFO_API_TOKEN")  # .env では GBIZINFO_API_TOKEN として定義

# --- 保存先パスの設定（絶対パスで迷子を防ぐ） ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_PATH = os.path.join(DATA_DIR, "raw_api_data.json")
CSV_PATH = os.path.join(DATA_DIR, "company.csv")

# ==========================================
# STEP 1: EDINETから売上ランキングデータを取得してJSON保存
# ==========================================
def step1_fetch_and_save_json():
    print("--- [STEP 1] EDINET APIから売上データを取得します ---")
    if not EDB_API_KEY:
        print("【エラー】.env に EDB_API_KEY が設定されていません。")
        return False

    API_URL = "https://edinetdb.jp/v1/rankings/revenue?limit=500"
    headers = {"X-API-KEY": EDB_API_KEY}
    
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code != 200:
            print(f"【エラー】API取得失敗: ステータスコード {response.status_code}")
            return False

        data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else data
        extracted_data = []

        # 「企業名」と「売上」を抽出
        for item in items:
            name = item.get("name")
            revenue = item.get("value")  
            extracted_data.append({"name": name, "revenue": revenue})

        # dataフォルダがなければ作成
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✨ JSONを保存しました: {JSON_PATH} ({len(extracted_data)}件)\n")
        return True

    except Exception as e:
        print(f"【通信エラー】APIへの接続中にエラーが発生しました: {e}")
        return False

# ==========================================
# STEP 2の補助関数: gBizINFOからURLを取得
# ==========================================
def get_url_from_gbizinfo(name):
    """gBizINFO APIを使って企業名からURLを取得する（2ステップ完全版）"""
    base_url = "https://info.gbiz.go.jp/hojin/v1/hojin"
    
    clean_api_key = GBIZ_API_KEY.strip() if GBIZ_API_KEY else ""
    clean_api_key = clean_api_key.replace('"', '').replace("'", "")
    
    headers = {
        "Accept": "application/json",
        "X-hojinInfo-api-token": clean_api_key
    }

    try:
        # 【ステップ1】企業名で検索し、法人番号（corporate_number）を取得
        search_response = requests.get(base_url, headers=headers, params={"name": name}, timeout=10)
        
        if search_response.status_code != 200:
            print(f" [検索エラー: {search_response.status_code}] ", end="")
            return "NOT_FOUND"
            
        search_data = search_response.json()
        hojin_list = search_data.get("hojin-infos", [])
        
        if not hojin_list:
            print(" [ヒットなし] ", end="")
            return "NOT_FOUND"
            
        # 完全一致する企業を優先して法人番号を特定
        target_hojin = next((h for h in hojin_list if h.get("name") == name), hojin_list[0])
        corp_number = target_hojin.get("corporate_number")
        
        if not corp_number:
            return "NOT_FOUND"

        # 【ステップ2】取得した法人番号を使って、詳細データ（URL）を取得
        detail_url = f"{base_url}/{corp_number}"
        detail_response = requests.get(detail_url, headers=headers, timeout=10)
        
        if detail_response.status_code != 200:
            print(f" [詳細取得エラー: {detail_response.status_code}] ", end="")
            return "NOT_FOUND"
            
        detail_data = detail_response.json()
        detail_hojin_list = detail_data.get("hojin-infos", [])
        
        if not detail_hojin_list:
            return "NOT_FOUND"
            
        # 詳細データの中から company_url を抽出
        company_url = detail_hojin_list[0].get("company_url")
        
        return company_url if company_url else "NOT_FOUND"

    except Exception as e:
        print(f" [例外エラー: {e}] ", end="")
        return "ERROR"
# ==========================================
# STEP 2: JSONデータとgBizINFOを組み合わせてCSVを作成
# ==========================================
def step2_generate_csv():
    print("--- [STEP 2] gBizINFO APIを使ってURLを取得し、CSVを作成します ---")

    if not GBIZ_API_KEY:
        print("【エラー】.env に GBIZ_API_KEY が設定されていません。")
        return

    if not os.path.exists(JSON_PATH):
        print(f"【エラー】{JSON_PATH} が見つかりません。先に Step 1 を実行してください。")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        companies = json.load(f)

    # 保存先フォルダの確認
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["company_name", "url", "revenue_class"])

        success_count = 0
        
        for i, comp in enumerate(companies):
            name = comp.get("name")
            revenue = comp.get("revenue")

            if not name or revenue is None:
                continue

            # 売上クラスの判定
            rev_num = int(revenue)
            if rev_num >= 2000000:
                rev_class = "S"
            elif rev_num >= 800000:
                rev_class = "A"
            elif rev_num >= 500000:
                rev_class = "B"
            else:
                rev_class = "C"

            print(f"[{i+1}/{len(companies)}] 取得中: {name} ... ", end="", flush=True)
            
            url = get_url_from_gbizinfo(name)
            
            if url and url != "NOT_FOUND" and url != "ERROR":
                print("OK")
                writer.writerow([name, url, rev_class])
                success_count += 1
            else:
                print("URLなし")
                writer.writerow([name, "NOT_FOUND", rev_class])
                
            # gBizINFO APIのRate Limit対策 (1秒間に10リクエストまで)
            time.sleep(0.5)

    print(f"\n✨ 処理完了: {success_count} 件のURLを取得し、CSVに保存しました！")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    # step1_fetch_and_save_json()
    step2_generate_csv()