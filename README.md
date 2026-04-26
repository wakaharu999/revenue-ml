## 売り上げ予測AI
企業の売り上げを企業HPのテキストから予測します

| クラス | 売上レンジ |
|--------|------------|
| S | 10兆円〜 |
| A | 1兆〜10兆円 |
| B | 5000億〜1兆円 |
| C | 〜5000億円 |

## データ収集
## 特徴量エンジニアリング
- **テキスト埋め込みベクトル**
    - テキストエンコーダーモデル：`multilingual-e5-base`
    - 処理フロー: ページ種別ごとのエンコード→ベクトルの結合
    - 意図と仮説
        - 企業のグローバル化による英語表現の増加：英語表現がテキストないで重要なトークンであったときに日本語のモデル`bert-base-japanese-v3`においては逆にノイズになってしまうのではないか。
        - 事前学習方法の違い：BERTなどのMLMは、事前学習の手法がマスクしたトークンを当てる自己教師あり学習にのみ基づいていることから、テキストの意味・文脈を埋め込むことに適していると考えられるが、今回のような分類タスクにおいてはファインチューニングが前提である。一方で、E5モデルの事前学習方法は、弱教師あり学習による対照学習に基づいており、事業領域・方針など売り上げ予測において重要な要素を意味空間上でしっかりと区別することができるのではないか。
        - `multilingual-e5-base`の評価結果を示すテクニカルレポートにおける日本語の検索能力の評価は非常に高く、多言語だからといって日本語への精度が劣るわけではないから[@wang2024multilinguale5textembeddings]。
       
- **追加特徴量**
    - 詳細を`/notebook/eda.ipynb`に記載(キーワード特徴量・TF-IDF特徴量)
## モデル設計
- **モデル概要**
    - 入力：テキストベクトル＋正規表現によるキーワード特徴＋構造特徴量
    - 出力：売り上げクラス

    - ネットワーク構成

    | セクション | 処理内容 |
    | :--- | :--- |
    | テキストMLP | Dense(256) → BN → Dropout(0.3) → Dense(64) → BN → Dropout(0.3) |
     キーワード | Dense(64) → BN → Dropout(0.2) |
    |結合層 | 両MLPの出力を結合|
    | 共通MLP | Dense(256) → BN → Dropout(0.4) → Dense(classes) |

    - 活性化関数:
        - 隠れ層：`ReLU` 
        - 出力層：`Softmax`
    - 過学習防止のため、各層に BatcNormalization と Dropout を適用した
- **設計意図**
    - 入力をテキスト埋め込みベクトル＋キーワード・構造特徴にしたのは`Silhouette Score`が優れていたEDAの結果を踏またから。
    - それぞれの特徴に対して全結合層を挟んでから結合を行ったのは次元数などのモダリティの違いからいきなり結合すると学習が壊れるのではないかと判断したため。
    - テキストベクトルのみを用いたTF-IDFを使ったゲーティング処理によるテキストベクトルでも同程度のいい結果が得られたが(`/model/model_tuning.ipynb`)、推論精度のブレイクスルーが見られなかったことと、トークン辞書作成はサンプルデータへの依存性が高く、今回サンプルデータに含まなかった世界の企業やベンチャー企業と辞書との不一致を踏まえAPIとしての機能性を考慮し採用しなかった。

- **採用モデルの交差検証の結果**
    - **各評価指標値**
      - Average Accuracy: 0.4039 (± 0.0629)
      - Macro F1 Score  : 0.3016
        | クラス | precision | recall | f1-score | support |
        | :--- | :---: | :---: | :---: | :---: |
        | A | 0.50 | 0.54 | 0.52 | 153 |
        | B | 0.34 | 0.45 | 0.39 | 138 |
        | C | 0.37 | 0.26 | 0.30 | 145 |
        | S | 0.00 | 0.00 | 0.00 | 12 |
        | | | | | |
        | **accuracy** | | | 0.40 | 448 |
        | **macro avg** | 0.30 | 0.31 | 0.30 | 448 |
        | **weighted avg** | 0.39 | 0.40 | 0.39 | 448 |

    - **混同行列**
  
    ![混同行列](/data/result_1.png)

## テスト結果
- **三菱商事株式会社(正解ランクS)**
```json
{
  "url": "https://www.mitsubishicorp.com/jp/ja/",
  "estimated_revenue_class": "A",
  "estimated_revenue_range": "売り上げ1兆円以上10兆円未満",
  "confidence": 0.8601,
  "class_probabilities": {
    "A": 0.8601,
    "B": 0.009,
    "C": 0.074,
    "S": 0.057
  },
  "features_summary": {
    "pages_crawled": 4,
    "has_ir_page": true,
    "has_recruit_page": false,
    "text_length_total": 4709
  },
  "processing_time_sec": 6.88
}
```

- **日産自動車株式会社(正解ランクS)**
```json
{
  "url": "https://www.nissan-global.com/JP/",
  "estimated_revenue_class": "C",
  "estimated_revenue_range": "売り上げ5000億円未満",
  "confidence": 0.5108,
  "class_probabilities": {
    "A": 0.0287,
    "B": 0.4467,
    "C": 0.5108,
    "S": 0.0138
  },
  "features_summary": {
    "pages_crawled": 7,
    "has_ir_page": true,
    "has_recruit_page": true,
    "text_length_total": 5162
  },
  "processing_time_sec": 7.63
}
```

- **株式会社セブン＆アイ・ホールディングス(正解ランクA)**
```json
{
  "url": "https://www.7andi.com/",
  "estimated_revenue_class": "A",
  "estimated_revenue_range": "売り上げ1兆円以上10兆円未満",
  "confidence": 0.7845,
  "class_probabilities": {
    "A": 0.7845,
    "B": 0.0415,
    "C": 0.1119,
    "S": 0.0621
  },
  "features_summary": {
    "pages_crawled": 5,
    "has_ir_page": true,
    "has_recruit_page": false,
    "text_length_total": 4288
  },
  "processing_time_sec": 6.68
}
```

- **株式会社SUBARU(正解ランクA)**
```json
{
  "url": "https://www.subaru.co.jp/",
  "estimated_revenue_class": "C",
  "estimated_revenue_range": "売り上げ5000億円未満",
  "confidence": 0.4897,
  "class_probabilities": {
    "A": 0.3988,
    "B": 0.0612,
    "C": 0.4897,
    "S": 0.0502
  },
  "features_summary": {
    "pages_crawled": 5,
    "has_ir_page": true,
    "has_recruit_page": true,
    "text_length_total": 6548
  },
  "processing_time_sec": 9.51
}
```

- **株式会社フジ・メディア・ホールディングス(正解ランクB)**
```json
{
  "url": "https://www.fujimediahd.co.jp/",
  "estimated_revenue_class": "C",
  "estimated_revenue_range": "売り上げ5000億円未満",
  "confidence": 0.6741,
  "class_probabilities": {
    "A": 0.0968,
    "B": 0.2012,
    "C": 0.6741,
    "S": 0.0279
  },
  "features_summary": {
    "pages_crawled": 6,
    "has_ir_page": true,
    "has_recruit_page": true,
    "text_length_total": 12028
  },
  "processing_time_sec": 12.36
}
```

- **日本テレビホールディングス株式会社(正解ランクC)**
```json
{
  "url": "https://www.ntvhd.co.jp/",
  "estimated_revenue_class": "A",
  "estimated_revenue_range": "売り上げ1兆円以上10兆円未満",
  "confidence": 0.781,
  "class_probabilities": {
    "A": 0.781,
    "B": 0.0146,
    "C": 0.1467,
    "S": 0.0577
  },
  "features_summary": {
    "pages_crawled": 5,
    "has_ir_page": false,
    "has_recruit_page": true,
    "text_length_total": 842
  },
  "processing_time_sec": 4.36
}
```
## 振り返り
