import scrapy # type: ignore

class CorpSpiderItem(scrapy.Item):
    # 以下の4つが「記入欄」です。一字一句間違えないようにしてください。
    url = scrapy.Field()
    revenue_class = scrapy.Field()
    page_title = scrapy.Field()
    text_content = scrapy.Field()