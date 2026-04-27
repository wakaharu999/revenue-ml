# Apple Silicon (arm64) で安定する軽量Pythonイメージ
FROM python:3.10-slim

WORKDIR /app

# ログのリアルタイム出力と、TensorFlowの不要な警告をミュート
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# ビルド時に必要なパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# E5モデルをダウンロードし、./models/multilingual-e5-base フォルダに保存して同梱する
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('intfloat/multilingual-e5-base'); \
model.save('./models/multilingual-e5-base')\
"

# 「COPY . .」をやめて、本番APIに必要なものだけを厳選してコピー
COPY src/ ./src/
COPY models/ ./models/
# ※もしクローリングや推論でdataフォルダの一部が必要なら、以下のコメントアウトを外してください
# COPY data/ ./data/

# コンテナ起動時にAPIサーバー（FastAPI）を立ち上げ
# host 0.0.0.0 は、コンテナ外部からアクセス可能にするための設定
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
