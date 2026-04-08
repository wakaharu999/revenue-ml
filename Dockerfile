# ベースとしてTensorFlow公式のイメージを使用
FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# コンテナ起動時にAPIサーバー（FastAPI）を立ち上げ
CMD ["uvicorn", "main.app", "--host", "0.0.0.0", "--port", "8080"]