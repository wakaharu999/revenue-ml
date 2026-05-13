import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import scrapy

CATEGORIES = ['top', 'about', 'history', 'business', 'ir', 'recruit', 'news', 'sustainability']
class RevenueSpider(scrapy.Spider):
    name = "revenue_spider"
    
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'DEPTH_LIMIT': 3,
        'FEED_EXPORT_ENCODING': 'utf-8',
        'BOT_NAME': 'company_crawler',
        'SPIDER_MODULES': ['company_crawler.spiders'],
        'NEWSPIDER_MODULE': 'company_crawler.spiders',
    
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4,
        'DOWNLOAD_DELAY': 1.0,
        'DOWNLOAD_TIMEOUT': 15,
        'CLOSESPIDER_TIMEOUT': 280,
        'LOG_LEVEL': 'ERROR',
    }

    def __init__(self, start_url=None, temp_file=None, *args, **kwargs):
        super(RevenueSpider, self).__init__(*args, **kwargs)
        
        domain = urlparse(start_url).netloc if start_url else ""
        self.start_requests_custom = [
            scrapy.Request(
                url=start_url, 
                meta={'category': 'top', 'allowed_domain': domain}
            )
        ] if start_url else []
        
        # 🌟 result_queue を temp_file に変更し、found_categories を削除
        self.temp_file = temp_file
        self.collected_texts = {cat: "" for cat in CATEGORIES}
        self.pages_crawled = 0

    def start_requests(self):
        for req in self.start_requests_custom:
            yield req

    def parse(self, response):
        self.pages_crawled += 1
        category = response.meta.get('category', 'top')
        allowed_domain = response.meta.get('allowed_domain', '')

        if hasattr(response, 'text') == False:
            return
        
        # 1. HTMLから人間が読めるテキストだけを綺麗に抽出する
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "header", "footer", "nav", "noscript", "aside", "form", "iframe"]):
            element.decompose()
        for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', text)

        # 2. テキストの保存（短すぎる場合はノイズとして捨てる）
        # 2. テキストの保存
        if len(clean_text) > 50:
            if len(self.collected_texts[category]) < 5000:
                self.collected_texts[category] += " " + clean_text[:5000]

        # 🌟 3. ページ内のリンクを探して辿る（if category == 'top': を削除！）
        for a_tag in response.css('a'):
            link_url = a_tag.attrib.get('href')
            link_text = a_tag.css('::text').get(default='').strip().lower()

            if not link_url:
                continue

            parsed_link = urlparse(link_url.lower())
            if re.search(r'\.(pdf|zip|doc|docx|xls|xlsx|png|jpg|jpeg|gif)$', parsed_link.path):
                continue

            absolute_url = response.urljoin(link_url)
            
            # 自分と同じドメインのリンクしか辿らない
            if allowed_domain not in urlparse(absolute_url).netloc:
                continue

            # 4. URLやリンクの文字列からカテゴリを判定する
            next_category = self._categorize_link(absolute_url.lower(), link_text)
            
            # 🌟 まだ文字数が5000字に達していないカテゴリならリンクを辿る
            if next_category:
                if len(self.collected_texts[next_category]) < 5000:
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse,
                        meta={'category': next_category, 'allowed_domain': allowed_domain}
                    )

    def _categorize_link(self, url, text):
        # 判定漏れを防ぐため、念のためURLとテキストを小文字化しておく
        url = url.lower()
        text = text.lower()

        # 1. 企業情報・理念・トップメッセージ (大幅強化)
        if re.search(r'about|company|profile|corporate|message|philosophy|vision', url) or \
           re.search(r'会社|企業|概要|理念|ビジョン|ミッション|メッセージ|ご挨拶|トップ|代表', text):
            return 'about'

        # 2. 事業・製品・サービス・実績 (大幅強化)
        elif re.search(r'business|service|solution|product|works|case', url) or \
             re.search(r'事業|サービス|ソリューション|製品|プロダクト|実績|事例|強み', text):
            return 'business'

        # 3. サステナビリティ・CSR・SDGs (✨新規追加: 大企業ほど情報量が多い=売上予測に超重要)
        elif re.search(r'sustainability|csr|esg|sdgs|environment', url) or \
             re.search(r'サステナビリティ|環境|社会|ガバナンス|sdgs|csr', text):
            return 'sustainability'

        # 4. IR・財務・投資家向け
        elif re.search(r'ir|investor|finance|highlight', url) or \
             re.search(r'ir|投資家|財務|業績|決算|ハイライト|株主', text):
            return 'ir'

        # 5. 採用・求人・働く環境
        elif re.search(r'recruit|career|jobs|採用|求人|エントリー', url) or \
             re.search(r'採用|求人|キャリア|働く|新卒|中途', text):
            return 'recruit'

        # 6. 沿革・歴史
        elif re.search(r'history|沿革|歴史|歩み|創業', url) or \
             re.search(r'沿革|歴史|歩み|創業', text):
            return 'history'

        # 7. ニュース・プレスリリース
        elif re.search(r'news|press|release|info|topics', url) or \
             re.search(r'ニュース|お知らせ|プレスリリース|トピックス|最新情報', text):
            return 'news'
            
        return None

    def closed(self, reason):
        import json
        # 🌟 Queueへのputをやめて、一時ファイルに書き出す
        if self.temp_file:
            data = {
                'texts': self.collected_texts,
                'summary': {
                    'pages_crawled': self.pages_crawled,
                    'has_ir_page': len(self.collected_texts['ir']) > 0,
                    'has_recruit_page': len(self.collected_texts['recruit']) > 0,
                    'text_length_total': sum(len(t) for t in self.collected_texts.values())
                }
            }
            with open(self.temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)