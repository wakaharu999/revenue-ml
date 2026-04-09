import scrapy # type: ignore
import csv
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup # type: ignore
from company_crawler.items import CompanyPageItem # type: ignore

class CorporateSpider(scrapy.Spider):
    name = 'corporate'
    
    # 読み込むCSVパス
    CSV_PATH = '/app/data/train.csv'

    def start_requests(self):
        # CSVから企業データを読み込み、最初のリクエスト（トップページ）を作成
        with open(self.CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                company_name = row['company_name']
                url = row['url']
                revenue_class = row['revenue_class']

                if not url or url == 'NOT_FOUND':
                    continue
                
                # ドメインを抽出（同じドメイン内だけをクローリングするため）
                domain = urlparse(url).netloc

                yield scrapy.Request(
                    url=url,
                    callback=self.parse_page,
                    meta={
                        'company_name': company_name,
                        'revenue_class': revenue_class,
                        'allowed_domain': domain,
                        'category': 'top' # 最初はトップページ
                    }
                )

    def parse_page(self, response):
        # 1. HTMLから人間が読めるテキストだけを綺麗に抽出する
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 不要なタグ（スクリプト、スタイル、ヘッダー、フッターなど）を削除
        for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            script.decompose()
            
        # テキストの抽出と空白の整形
        text = soup.get_text(separator=' ', strip=True)
        # 連続する空白を1つにまとめる
        clean_text = re.sub(r'\s+', ' ', text)

        # 2. Itemの作成と保存（テキストが短すぎる場合はノイズとして捨てる）
        if len(clean_text) > 50:
            item = CompanyPageItem()
            item['company_name'] = response.meta['company_name']
            item['revenue_class'] = response.meta['revenue_class']
            item['url'] = response.url
            item['page_category'] = response.meta['category']
            item['text_content'] = clean_text
            yield item

        # 3. ページ内のリンクを探して辿る（トップページから来た時のみ）
        if response.meta['category'] == 'top':
            # ページ内のすべての <a> タグを取得
            for a_tag in response.css('a'):
                link_url = a_tag.attrib.get('href')
                link_text = a_tag.css('::text').get(default='').strip().lower()

                if not link_url:
                    continue

                # 相対パスを絶対URLに変換
                absolute_url = response.urljoin(link_url)
                
                # 自分と同じドメインのリンクしか辿らない
                if response.meta['allowed_domain'] not in urlparse(absolute_url).netloc:
                    continue

                # 4. URLやリンクの文字列から「どのカテゴリのページか」を判定する
                category = self._categorize_link(absolute_url.lower(), link_text)
                
                if category:
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse_page,
                        meta={
                            'company_name': response.meta['company_name'],
                            'revenue_class': response.meta['revenue_class'],
                            'allowed_domain': response.meta['allowed_domain'],
                            'category': category
                        }
                    )

    def _categorize_link(self, url, text):
        """URLやリンクテキストからページの種類を判定する"""
        if re.search(r'ir|investor|投資家|財務', url) or re.search(r'ir|投資家', text):
            return 'ir'
        elif re.search(r'recruit|career|採用|求人|エントリー', url) or re.search(r'採用|求人|キャリア', text):
            return 'recruit'
        elif re.search(r'about|company|profile|corporate|会社|概要', url) or re.search(r'会社|企業|概要', text):
            return 'about'
        elif re.search(r'history|沿革|歴史|歩み', url) or re.search(r'沿革|歴史', text):
            return 'history'
        elif re.search(r'business|service|solution|事業|サービス', url) or re.search(r'事業|サービス', text):
            return 'business'
        elif re.search(r'news|press|release|info|ニュース|お知らせ', url) or re.search(r'ニュース|プレスリリース', text):
            return 'news'
        return None # どれにも当てはまらないページは辿らない