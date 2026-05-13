import requests
from bs4 import BeautifulSoup
import csv
import time

def scrape_j_startup_details():
    base_url = "https://www.j-startup.go.jp"
    list_url = f"{base_url}/startups/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print("1. 企業一覧ページから詳細ページのURLリストを取得中...")
    try:
        response = requests.get(list_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        detail_links = []
        for a_tag in soup.select("a[href^='/startups/']"):
            href = a_tag.get("href")
    
            if isinstance(href, str) and href.endswith(".html") and href not in detail_links:
                detail_links.append(base_url + href if href.startswith("/") else href)

        detail_links = detail_links[:100]
        print(f"-> {len(detail_links)}件の詳細ページURLを取得しました。\n")

        print("2. 各詳細ページから企業名と公式URLを取得し、CSVに追記します...")
        
        with open("data/company.csv", mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            count = 0

            for detail_url in detail_links:
                time.sleep(1) 

                try:
                    res = requests.get(detail_url, headers=headers)
                    res.raise_for_status()
                    detail_soup = BeautifulSoup(res.content, "html.parser")

        
                    title_text = detail_soup.title.get_text() if detail_soup.title else ""
                    company_name = title_text.replace("｜J-Startup", "").strip()

                    company_url = ""
                    
                    # btm-arrow-blank というクラスを持つdivの中の <a> タグを直接指定して取得
                    link_tag = detail_soup.select_one("div.btm-arrow-blank a")
                    
                    if link_tag:
                        href = link_tag.get("href")
                        if isinstance(href, str):
                            company_url = href

                    if company_name and company_url:
                        revenue_class = "D"
                        writer.writerow([company_name, company_url, revenue_class])
                        print(f" 取得成功: {company_name} | {company_url}")
                        count += 1
                    else:
                        print(f" URLが見つかりませんでした: {detail_url}")

                except Exception as e:
                    print(f" 個別ページ取得エラー ({detail_url}): {e}")

        print(f"\n 完了: 合計 {count} 社のデータを company.csv に追記しました。")

    except Exception as e:
        print(f" 一覧ページ取得エラー: {e}")

if __name__ == "__main__":
    scrape_j_startup_details()