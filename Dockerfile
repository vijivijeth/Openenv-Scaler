FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
