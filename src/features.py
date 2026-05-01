import os
import re
import numpy as np
from multiprocessing import Process, Queue
from sentence_transformers import SentenceTransformer

from crawler import run_spider, CATEGORIES

# 警告ログのミュート
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 特徴量抽出のメインクラス
# ==========================================
class FeatureExtractor:
    def __init__(self):
        print("Loading SentenceTransformer (multilingual-e5-base) from local...")
        local_model_path = "./models/multilingual-e5-base" 
        self.model = SentenceTransformer(local_model_path)

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
        struct_features.append(np.log1p(len(re.findall(r'グローバル|global|海外|世界|多国籍|現地|国際|輸出|輸入|北米|アメリカ|欧州|ヨーロッパ|中国|ASEAN|アジア|中東|アフリカ|オセアニア', all_text))))
        struct_features.append(self._get_first_num(r'海外売上(?:高|比率).*?([0-9\.]+)%', all_text))
        struct_features.append(self._get_first_num(r'(?:海外|世界)(?:拠点)?[\s\w]{0,10}?([0-9,]+)(?:ヶ所|箇所|拠点)', all_text))
        struct_features.append(np.log1p(len(re.findall(r'明治|大正|昭和|平成|令和', all_text))))
        struct_features.append(self._get_first_num(r'(?:創業|創立|設立|発祥|始業).*?((?:18|19|20)[0-9]{2})年', all_text))
        struct_features.append(np.log1p(len(re.findall(r'M&A|買収|合併|統合|提携|アライアンス|alliance|パートナー|partner|協業|研究|産学|R&D|共同研究|イノベーション|innovation|特許|知的財産|大学|TOB|子会社|グループ会社|プライム', all_text))))
        struct_features.append(self._get_first_num(r'(?:連結子会社|グループ会社|関係会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text))
        struct_features.append(self._get_first_num(r'(?:提携先|パートナー数|協力会社)[\s\w:：]{0,10}?([0-9,]+)社', all_text))
        struct_features.append(np.log1p(len(re.findall(r'最大手|トップ|top|No\.1|ナンバーワン|リード|リーディングカンパニー|唯一|独自|オンリー|特許|only|best|首位', all_text))))
        struct_features.append(self._get_first_num(r'(?:国内|世界|市場)シェア.*?([0-9\.]+)%', all_text))
        struct_features.append(len(set(re.findall(r'([一-龥]{2,6}事業)', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'兆円|数兆|兆', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'億円|数百億|億', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'福利厚生|ワークライフバランス|くるみん|ホワイト|テレワーク|フレックス|ダイバーシティ|女性|育児|介護|休暇|健康|QOL|健康|DX', all_text))))
        struct_features.append(self._get_first_num(r'(?:連結|グループ|正社員|就業|全社)?(?:従業員|人員|社員|スタッフ)(?:数|合計)?[\s\w:：]{0,10}?([0-9,]+)(?:名|人)', all_text))
        struct_features.append(len(set(re.findall(r'([一-龥]{2,5}職)', all_text))))
        struct_features.append(np.log1p(len(re.findall(r'コーポレートガバナンス|コンプライアンス|内部統制|リスクマネジメント|監査|サステナビリティ|SDGs|ESG|CSR|環境保全|社会貢献|カーボンニュートラル|脱炭素', all_text))))

        # 構造＋キーワード
        final_struct_vector = np.concatenate([np.array(text_features_list), np.array(struct_features)]) 
        return {
            'text': final_text_vector,
            'struct': final_struct_vector
        }, summary