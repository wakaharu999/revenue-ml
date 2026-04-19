# Apple Silicon (arm64) で安定する軽量Pythonイメージ
FROM python:3.10-slim

WORKDIR /app

# ログのリアルタイム出力と、TensorFlowの不要な警告をミュートする魔法の呪文
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 必要なシステムパッケージ（curlはローカルでAPIを叩くテスト用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 「COPY . .」をやめて、本番APIに必要なものだけを厳選してコピー
COPY src/ ./src/
COPY models/ ./models/
# ※もしクローリングや推論でdataフォルダの一部が必要なら、以下のコメントアウトを外してください
# COPY data/ ./data/

# コンテナ起動時にAPIサーバー（FastAPI）を立ち上げ
# （srcディレクトリの中のmain.pyを起動する指定に変更）
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
