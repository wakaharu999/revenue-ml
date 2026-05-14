FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3'); \
AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-v3')\
"

COPY src/ ./src/
COPY models/ ./models/
COPY data/label_mappings.json ./data/label_mappings.json

# コンテナ起動時にAPIサーバー（FastAPI）を立ち上げ
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]