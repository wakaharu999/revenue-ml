import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import scrapy

CATEGORIES = ['top', 'about', 'history', 'business', 'ir', 'recruit', 'news']

class RevenueSpider(scrapy.Spider):
    name = "revenue_spider"
    
    custom_settings = {
        'USER_AGENT': 'CompanyInfoCrawler (+https://wakaharu999.com)',
        'DEPTH_LIMIT': 2,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'BOT_NAME': 'company_crawler',
        'SPIDER_MODULES': ['company_crawler.spiders'],
        'NEWSPIDER_MODULE': 'company_crawler.spiders',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'DOWNLOAD_DELAY': 3.0,
        
        # APIとして途中で落ちないようタイムアウトを延長
        'DOWNLOAD_TIMEOUT': 10,
        'CLOSESPIDER_TIMEOUT': 45,
        'LOG_LEVEL': 'ERROR',
    }

    def __init__(self, start_url=None, result_queue=None, *args, **kwargs):
        super(RevenueSpider, self).__init__(*args, **kwargs)
        
        # ドメインを抽出してメタデータに保存（source 1のロジック）
        domain = urlparse(start_url).netloc if start_url else ""
        self.start_requests_custom = [
            scrapy.Request(
                url=start_url, 
                meta={'category': 'top', 'allowed_domain': domain}
            )
        ] if start_url else []
        
        self.result_queue = result_queue
        self.collected_texts = {cat: "" for cat in CATEGORIES}
        self.pages_crawled = 0
        self.found_categories = set(['top'])

    def start_requests(self):
        for req in self.start_requests_custom:
            yield req

    def parse(self, response):
        self.pages_crawled += 1
        category = response.meta.get('category', 'top')
        allowed_domain = response.meta.get('allowed_domain', '')
        
        # 1. HTMLから人間が読めるテキストだけを綺麗に抽出する
        soup = BeautifulSoup(response.body, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', text)

        # 2. テキストの保存（短すぎる場合はノイズとして捨てる）
        if len(clean_text) > 50:
            if len(self.collected_texts[category]) < 7000:
                self.collected_texts[category] += " " + clean_text[:7000]

        # 3. ページ内のリンクを探して辿る（トップページから来た時のみ）
        if category == 'top':
            for a_tag in response.css('a'):
                link_url = a_tag.attrib.get('href')
                link_text = a_tag.css('::text').get(default='').strip().lower()

                if not link_url:
                    continue

                absolute_url = response.urljoin(link_url)
                
                # 自分と同じドメインのリンクしか辿らない
                if allowed_domain not in urlparse(absolute_url).netloc:
                    continue

                # 4. URLやリンクの文字列からカテゴリを判定する
                next_category = self._categorize_link(absolute_url.lower(), link_text)
                
                if next_category and next_category not in self.found_categories:
                    self.found_categories.add(next_category)
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse,
                        meta={'category': next_category, 'allowed_domain': allowed_domain}
                    )

    def _categorize_link(self, url, text):
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
        return None

    def closed(self, reason):
        # 収集したテキストをQueueに返す
        self.result_queue.put({ # type: ignore
            'texts': self.collected_texts,
            'summary': {
                'pages_crawled': self.pages_crawled,
                'has_ir_page': len(self.collected_texts['ir']) > 0,
                'has_recruit_page': len(self.collected_texts['recruit']) > 0,
                'text_length_total': sum(len(t) for t in self.collected_texts.values())
            }
        })