# Revenue Range Classifier
## プロジェクト概要
- 企業のWebサイトのテキストデータから売り上げ規模を5つのクラスに分類する深層学習プロジェクトです。
- 本モデルはREST APIとして公開されており、Swagger UI (/docs) から直接テストが可能です。
- **URL:** https://haru-first-app-19443304909.asia-northeast1.run.app/docs
    - **リクエスト：**企業URL
    - **レスポンス：**分類結果
- プロジェクトの動機・目的
| クラス | 売上レンジ |
|--------|------------|
| S | 2兆円〜 |
| A | 8000億〜2兆円 |
| B | 5000億〜8000億円 |
| C | 〜5000億円 |
| D | スタートアップ企業|

## プロジェクトの動機・目的

## アーキテクチャ概要
### 推論フロー
1. リクエスト：ユーザーが`/estimate`エンドポイントにURLを送信する。
2. クローリング：`RevenueSpider`がWebサイトのテキストコンテンツを取得し、内部のカテゴリ分類ロジックを用いて価値のリンクを巡回する。
3. 特徴抽出：
4. クラス分類：
5. レスポンス：
### 推論モデルアーキテクチャ

## ディレクトリ構成

## データ概要
- EDINET DB(https://edinetdb.jp/)における売上ランキングトップ500の企業のHPをクローリングを行って得たテキスト群
- J-Startup(https://www.j-startup.go.jp/startups/)に登録されている企業の内100社を対象にクローリングして得たテキスト群

##　技術スタック
- **Machine Learning:** PyTorch, Transformers(Hugging Face)
- **Backend / API:** FastAPI, Uvicorn, Pydantic
- **Infrastructure:** Docker, Google Cloud Run
- **Data Engineering:** Pandas, NumPy
- **Web Crawling:**: Scrapy

##　ドキュメント概要

