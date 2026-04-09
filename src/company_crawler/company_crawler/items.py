# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy # type: ignore

class CompanyPageItem(scrapy.Item):
    company_name = scrapy.Field()
    revenue_class = scrapy.Field()
    url = scrapy.Field()
    page_category = scrapy.Field() # top, about, history, business, ir, recruit, news など
    text_content = scrapy.Field()