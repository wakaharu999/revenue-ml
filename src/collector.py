import os
import json
import requests # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
API_KEY = os.getenv("EDB_API_KEY")

# 保存先のファイルパス
JSON_PATH = "/app/data/raw_api_data.json"

# ==========================================
# STEP 1: APIからデータを取得してJSONに保存
# ==========================================
def step1_fetch_and_save_json():
    print("--- [STEP 1] APIからデータを取得します ---")
    if not API_KEY:
        print("【エラー】EDINET_API_KEYが設定されていません。")
        return False

    # 100件取得するURL
    API_URL = "https://edinetdb.jp/v1/rankings/revenue?limit=500"
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

        # APIのレスポンスから「企業名」と「売上」だけを抽出
        for item in items:
            name = item.get("name")
            revenue = item.get("value")  # 売上の値
            
            # データが存在する企業のみリストに追加
            if name and revenue:
                extracted_data.append({
                    "name": name,
                    "revenue": revenue
                })

        # JSONファイルとして保存（dataフォルダが無ければ作る）
        os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSONを保存しました: {JSON_PATH} ({len(extracted_data)}件)")
        return True

    except Exception as e:
        print(f"【通信エラー】APIへの接続中にエラーが発生しました: {e}")
        return False

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    step1_fetch_and_save_json()