import scrapy # type: ignore
import csv  # CSVファイルを読み込むための道具
from corp_spider.corp_spider.items import CorpSpiderItem

class CorpInfoSpider(scrapy.Spider):
    name = "corp_info_spider"

    # 1. 調査の「出発地点」を自分で決める関数
    def start_requests(self):
        # コンテナ内の /app/data/train.csv を開く
        with open('/app/data/train.csv', 'r', encoding='utf-8') as f:
            # CSVの中身を一行ずつ辞書形式（列名がキーになる）で読み込む
            reader = csv.DictReader(f)
            for row in reader:
                url = row['url']
                # インターネットへ出発！
                # metaを使うと、あとの処理（parse）に「売上ランク」などを持ち越せます
                yield scrapy.Request(
                    url=url, 
                    callback=self.parse, 
                    meta={'revenue_class': row['revenue_class'], 'company_name': row['company_name']}
                )

    # 2. ページが読み込まれたら実行される関数
    def parse(self, response):
        item = CorpSpiderItem()
        
        # 預けていた meta 情報を回収して記録用紙（Item）に書く
        item['url'] = response.url
        item['revenue_class'] = response.meta['revenue_class']
        
        # ページのタイトルを抜き出す
        item['page_title'] = response.css('title::text').get()
        
        # ページ内の段落（pタグ）の文字を全部つなげて取得
        texts = response.css('p::text').getall()
        item['text_content'] = " ".join(texts).strip()

        # 1社分の報告書を提出！
        yield item