import os
import csv
import sys
import json
import time
import multiprocessing
from urllib.parse import urlparse

# --- パスの設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "company.csv")
JSONL_PATH = os.path.join(BASE_DIR, "data", "crawled_data.jsonl")

def run_spider_process(start_url, queue):
    """
    別プロセスでScrapyクローラーを起動するための関数。
    """
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)

    from scrapy.crawler import CrawlerProcess
    from src.crawler import RevenueSpider

    try:
        # クローラーの起動
        process = CrawlerProcess({'LOG_LEVEL': 'ERROR'})
        process.crawl(RevenueSpider, start_url=start_url, result_queue=queue)
        process.start()
    except Exception as e:
        print(f"\n[Spider Error] {e}")

def main():
    print("--- 🕸️ クローリングを開始します ---")

    if not os.path.exists(CSV_PATH):
        print(f"【エラー】{CSV_PATH} が見つかりません。")
        return

    # すでにクローリング済みの企業を記録（途中で止まっても再開できるようにするため）
    crawled_companies = set()
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    crawled_companies.add(data["company_name"])
                except json.JSONDecodeError:
                    pass

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # JSONLファイルは追記モード('a')で開く
    with open(JSONL_PATH, "a", encoding="utf-8") as jsonl_file:
        for i, row in enumerate(rows):
            company_name = row.get("company_name", "")
            url = row.get("url", "")
            revenue_class = row.get("revenue_class", "")

            # URLが無効な場合や、既にクローリング済みの場合はスキップ
            if not url or url in ("NOT_FOUND", "ERROR"):
                continue
            if company_name in crawled_companies:
                print(f"[{i+1}/{len(rows)}] ⏩ スキップ (取得済): {company_name}")
                continue

            print(f"[{i+1}/{len(rows)}] 🔍 クローリング中: {company_name} ({url}) ... ", end="", flush=True)

            # プロセス間通信用のキューを作成
            queue = multiprocessing.Queue()
            
            # クローラーを別プロセスで実行
            p = multiprocessing.Process(target=run_spider_process, args=(url, queue))
            p.start()
            
            # crawler.py の CLOSESPIDER_TIMEOUT(45秒) + 余裕をもたせて最大60秒待機
            p.join(timeout=100)

            # もし60秒経ってもプロセスが終わっていなければ強制終了
            if p.is_alive():
                print("タイムアウト (強制終了)")
                p.terminate()
                p.join()
                continue

            # キューから結果を取得してJSONLに書き込む
            if not queue.empty():
                result = queue.get()
                texts_dict = result.get("texts", {})
                
                saved_categories = 0
                for category, text_content in texts_dict.items():
                    # テキストが空でなければ保存
                    clean_content = text_content.strip()
                    if clean_content:
                        json_record = {
                            "company_name": company_name,
                            "revenue_class": revenue_class,
                            "url": url,
                            "page_category": category,
                            "text_content": clean_content
                        }
                        # jsonl形式で1行ずつ書き込み
                        jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                        saved_categories += 1
                        
                print(f"完了 ({saved_categories} カテゴリ保存)")
            else:
                print("失敗 (データ取得できず)")

            time.sleep(2)

    print("\n🎉 すべてのクローリング処理が完了しました！")
    print(f"出力先: {JSONL_PATH}")

if __name__ == "__main__":
    # WindowsやmacOSでのmultiprocessingの安全な実行のためのおまじない
    multiprocessing.freeze_support()
    main()