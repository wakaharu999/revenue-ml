import os
import re
import numpy as np
from multiprocessing import Process, Queue
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import scrapy
from scrapy.crawler import CrawlerProcess

# 警告ログのミュート
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CATEGORIES = ['top', 'about', 'history', 'business', 'ir', 'recruit', 'news']

# ==========================================
# 1. API用のScrapyスパイダー（使い捨てクローラー）
# ==========================================
class RevenueSpider(scrapy.Spider):
    name = "revenue_spider"
    custom_settings = {
        'DOWNLOAD_TIMEOUT': 10,          # 1ページ10秒で諦める
        'CLOSESPIDER_TIMEOUT': 25,       # 全体25秒で強制終了
        'DEPTH_LIMIT': 1,                # リンクの深さ
        'LOG_LEVEL': 'ERROR',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    def __init__(self, start_url=None, result_queue=None, *args, **kwargs):
        super(RevenueSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url] if start_url else []
        self.result_queue = result_queue
        self.collected_texts = {cat: "" for cat in CATEGORIES}
        self.pages_crawled = 0

    def parse(self, response):
        self.pages_crawled += 1
        url = response.url.lower()
        
        # 本文抽出
        soup = BeautifulSoup(response.body, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        # URLからカテゴリ判定
        page_type = "top"
        if "about" in url or "company" in url: page_type = "about"
        elif "history" in url: page_type = "history"
        elif "business" in url or "service" in url: page_type = "business"
        elif "ir" in url or "investor" in url: page_type = "ir"
        elif "recruit" in url or "career" in url: page_type = "recruit"
        elif "news" in url or "press" in url: page_type = "news"

        self.collected_texts[page_type] += " " + text

        # Topページならリンクを辿る
        if page_type == "top":
            links = response.css('a::attr(href)').getall()
            for link in links[:8]: # 速度重視で最大8リンク
                yield response.follow(link, self.parse)

    def closed(self, reason):
        self.result_queue.put({ # type: ignore
            'texts': self.collected_texts,
            'summary': {
                'pages_crawled': self.pages_crawled,
                'has_ir_page': len(self.collected_texts['ir']) > 0,
                'has_recruit_page': len(self.collected_texts['recruit']) > 0,
                'text_length_total': sum(len(t) for t in self.collected_texts.values())
            }
        })

# Twisted(Reactor)の再起動エラーを防ぐための別プロセス起動関数
def run_spider(url, queue):
    process = CrawlerProcess()
    process.crawl(RevenueSpider, start_url=url, result_queue=queue)
    process.start()

# ==========================================
# 2. 特徴量抽出のメインクラス
# ==========================================
class FeatureExtractor:
    def __init__(self):
        print("Loading SentenceTransformer (multilingual-e5-base)...")
        # 学習時と全く同じモデル(768次元)を使用
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')

    # 数字抽出のヘルパー関数
    def _get_first_num(self, pattern, text, default=0):
        match = re.search(pattern, text)
        if match and match.group(1) is not None:
            num_str = re.sub(r'[^\d.]', '', match.group(1))
            try: return float(num_str)
            except ValueError: return default
        return default

    def extract_from_url(self, url: str):
        """APIから呼び出されるメインメソッド"""
        q = Queue()
        p = Process(target=run_spider, args=(url, q))
        p.start()
        
        # スパイダー側が25秒で終わるはずなので、30秒で強制回収
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()
            return None, None
        if q.empty():
            return None, None

        crawl_result = q.get()
        collected_texts = crawl_result['texts']
        summary = crawl_result['summary']

        all_text = " ".join(collected_texts.values())
        total_length = len(all_text) + 1 

        # --------------------------------------------------
        # 特徴量A: テキストベクトル側（5397次元）
        # (7カテゴリ×768次元) + (has, length, ratio × 7カテゴリ = 21次元) = 5397
        # --------------------------------------------------
        text_features_list = []
        text_vector_list = []

        for cat in CATEGORIES:
            cat_text = collected_texts.get(cat, "")
            
            # 1. 21次元分のメタデータ（has, length, ratio）
            text_features_list.append(1 if len(cat_text) > 0 else 0)
            text_features_list.append(np.log1p(len(cat_text)))
            text_features_list.append(len(cat_text) / total_length)

            # 2. E5モデルによるベクトル化（768次元）
            if not cat_text:
                text_vector_list.extend(np.zeros(768))
            else:
                chunks = [f"passage: {cat_text[i:i+400]}" for i in range(0, len(cat_text), 400)]
                embeddings = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
                pooled_vec = np.mean(embeddings, axis=0)
                text_vector_list.extend(pooled_vec)

        # 21 + 5376 = 5397次元
        final_text_vector = np.concatenate([np.array(text_features_list), np.array(text_vector_list)])

        # --------------------------------------------------
        # 特徴量B: 構造的特徴量側（17次元）
        # --------------------------------------------------
        struct_features = []
        # 1. グローバル (3)
        struct_features.append(np.log1p(len(re.findall(r'グローバル|global|海外|世界|多国籍|現地|国際|輸出|輸入|北米|アメリカ|欧州|ヨーロッパ|中国|ASEAN|アジア|中東|アフリカ|オセアニア', all_text))))
        struct_features.append(self._get_first_num(r'海外売上(?:高|比率).*?([0-9\.]+)%', all_text))
        struct_features.append(self._get_first_num(r'(?:海外|世界)(?:拠点)?[\s\w]{0,10}?([0-9,]+)(?:ヶ所|箇所|拠点)', all_text))
        # 2. 歴史 (2)
        struct_features.append(np.log1p(len(re.findall(r'明治|大正|昭和|平成|令和', all_text))))
        struct_features.append(self._get_first_num(r'(?:創業|創立|設立|発祥|始業).*?((?:18|19|20)[0-9]{2})年', all_text))
        # 3. M&A (3)
        struct_features.append(np.log1p(len(re.findall(r'M&A|買収|合併|統合|提携|アライアンス|alliance|パートナー|partner|協業|研究|産学|R&D|共同研究|イノベーション|innovation|特許|知的財産|大学|TOB|子会社|グループ会社|プライム', all_text))))
        struct_features.append(self._get_first_num(r'(?:連結子会社|グループ会社|関係会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text))
        struct_features.append(self._get_first_num(r'(?:提携先|パートナー数|協力会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text))
        # 4. シェア (2)
        struct_features.append(np.log1p(len(re.findall(r'最大手|トップ|top|No\.1|ナンバーワン|リード|リーディングカンパニー|唯一|独自|オンリー|特許|only|best|首位', all_text))))
        struct_features.append(self._get_first_num(r'(?:国内|世界|市場)シェア.*?([0-9\.]+)%', all_text))
        # 5. 規模 (3)
        struct_features.append(len(set(re.findall(r'([一-龥]{2,6}事業)', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'兆円|数兆|兆', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'億円|数百億|億', all_text))))
        # 6. 人事 (3)
        struct_features.append(np.log1p(len(re.findall(r'福利厚生|ワークライフバランス|くるみん|ホワイト|テレワーク|フレックス|ダイバーシティ|女性|育児|介護|休暇|健康|QOL|健康|DX', all_text))))
        struct_features.append(self._get_first_num(r'(?:連結|グループ|正社員|就業|全社)?(?:従業員|人員|社員|スタッフ)(?:数|合計)?[\s\w:：]{0,10}?([0-9,]+)(?:名|人)', all_text))
        struct_features.append(len(set(re.findall(r'([一-龥]{2,5}職)', all_text))))
        # 7. ガバナンス (1)
        struct_features.append(np.log1p(len(re.findall(r'コーポレートガバナンス|コンプライアンス|内部統制|リスクマネジメント|監査|サステナビリティ|SDGs|ESG|CSR|環境保全|社会貢献|カーボンニュートラル|脱炭素', all_text))))

        final_struct_vector = np.array(struct_features) # 17次元

        return {
            'text': final_text_vector,
            'struct': final_struct_vector
        }, summary