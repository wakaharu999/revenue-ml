import re
from bs4 import BeautifulSoup
import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
from queue import Queue
from typing import Optional

CATEGORIES = ['top', 'about', 'history', 'business', 'ir', 'recruit', 'news']

# ==========================================
# 1. API用のScrapyスパイダー（使い捨てクローラー）
# ==========================================
class RevenueSpider(scrapy.Spider):
    name = "revenue_spider"
    custom_settings = {
        'DOWNLOAD_TIMEOUT': 7,
        'CLOSESPIDER_TIMEOUT': 15,
        'DEPTH_LIMIT': 1,
        'LOG_LEVEL': 'ERROR',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    def __init__(self, start_url: str, result_queue: Queue, *args, **kwargs):
        super(RevenueSpider, self).__init__(*args, **kwargs)
        self.start_requests_custom = [scrapy.Request(url=start_url, meta={'category': 'top'})] if start_url else []
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
        
        soup = BeautifulSoup(response.body, 'html.parser')
        
        for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', text)

        if len(clean_text) > 50:
            if len(self.collected_texts[category]) < 7000:
                self.collected_texts[category] += " " + clean_text[:7000]

        if category == 'top':
            domain = urlparse(response.url).netloc
            
            for a_tag in response.css('a'):
                link_url = a_tag.attrib.get('href')
                link_text = a_tag.css('::text').get(default='').strip().lower()

                if not link_url:
                    continue

                absolute_url = response.urljoin(link_url)
                
                if domain not in urlparse(absolute_url).netloc:
                    continue

                next_category = self._categorize_link(absolute_url.lower(), link_text)
                
                if next_category and next_category not in self.found_categories:
                    self.found_categories.add(next_category)
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse,
                        meta={'category': next_category}
                    )

    def _categorize_link(self, url, text):
        """正規表現判定"""
        if re.search(r'ir|investor|投資家|財務', url) or re.search(r'ir|投資家', text): return 'ir'
        elif re.search(r'recruit|career|採用|求人|エントリー', url) or re.search(r'採用|求人|キャリア', text): return 'recruit'
        elif re.search(r'about|company|profile|corporate|会社|概要', url) or re.search(r'会社|企業|概要', text): return 'about'
        elif re.search(r'history|沿革|歴史|歩み', url) or re.search(r'沿革|歴史', text): return 'history'
        elif re.search(r'business|service|solution|事業|サービス', url) or re.search(r'事業|サービス', text): return 'business'
        elif re.search(r'news|press|release|info|ニュース|お知らせ', url) or re.search(r'ニュース|プレスリリース', text): return 'news'
        return None

    def closed(self, reason):
        self.result_queue.put({
            'texts': self.collected_texts,
            'summary': {
                'pages_crawled': self.pages_crawled,
                'has_ir_page': len(self.collected_texts['ir']) > 0,
                'has_recruit_page': len(self.collected_texts['recruit']) > 0,
                'text_length_total': sum(len(t) for t in self.collected_texts.values())
            }
        })

def run_spider(url, queue):
    process = CrawlerProcess()
    process.crawl(RevenueSpider, start_url=url, result_queue=queue)
    process.start()