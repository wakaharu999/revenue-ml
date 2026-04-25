import os
import re
import numpy as np
from multiprocessing import Process, Queue
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse

# 警告ログのミュート
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    def __init__(self, start_url=None, result_queue=None, *args, **kwargs):
        super(RevenueSpider, self).__init__(*args, **kwargs)
        # 初期リクエストでメタデータ（top）を渡す
        self.start_requests_custom = [scrapy.Request(url=start_url, meta={'category': 'top'})] if start_url else []
        self.result_queue = result_queue
        self.collected_texts = {cat: "" for cat in CATEGORIES}
        self.pages_crawled = 0
        
        # 修正点: API用安全装置: 各カテゴリ1ページ見つけたら満足する
        self.found_categories = set(['top']) 

    def start_requests(self):
        for req in self.start_requests_custom:
            yield req

    def parse(self, response):
        self.pages_crawled += 1
        category = response.meta.get('category', 'top')
        
        # ==========================================
        # 1. crawler.py と完全一致のテキスト抽出
        # ==========================================
        soup = BeautifulSoup(response.body, 'html.parser')
        
        for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', text)

        # テキストの保存（短すぎる場合はノイズとして捨てるのも crawler.py と同じ）
        if len(clean_text) > 50:
            # 修正点: テキストが長すぎてメモリ爆発するのを防ぐ（最大7000文字）
            if len(self.collected_texts[category]) < 7000:
                self.collected_texts[category] += " " + clean_text[:7000]

        # ==========================================
        # 2. crawler.py と完全一致のリンク巡回
        # ==========================================
        if category == 'top':
            domain = urlparse(response.url).netloc
            
            for a_tag in response.css('a'):
                link_url = a_tag.attrib.get('href')
                link_text = a_tag.css('::text').get(default='').strip().lower()

                if not link_url:
                    continue

                absolute_url = response.urljoin(link_url)
                
                # 自分と同じドメインのリンクしか辿らない
                if domain not in urlparse(absolute_url).netloc:
                    continue

                # crawler.pyの正規表現関数でカテゴリ判定
                next_category = self._categorize_link(absolute_url.lower(), link_text)
                
                # 修正点: API用安全装置: newsも拾うが、無限ループを防ぐため各カテゴリ1ページのみ辿る
                if next_category and next_category not in self.found_categories:
                    self.found_categories.add(next_category)
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse,
                        meta={'category': next_category}
                    )

    def _categorize_link(self, url, text):
        """正規表現判定"""
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
        
        # スパイダー側が15秒で終わるはずなので、20秒で強制回収
        p.join(timeout=20)
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

        # HF埋め込み
        final_text_vector = np.array(text_vector_list)

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

        # 構造＋キーワード
        final_struct_vector = np.concatenate([np.array(text_features_list), np.array(struct_features)]) # テキスト側の21次元も結合して38次元に
        return {
            'text': final_text_vector,
            'struct': final_struct_vector
        }, summary