FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY chatbot_backend_claude_1.py /app/chatbot_backend.py

RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    gunicorn \
    anthropic \
    sentence-transformers \
    scikit-learn \
    numpy \
    scipy

ENV PORT=5003
ENV ANTHROPIC_API_KEY=""

EXPOSE 5003

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} chatbot_backend:app"]
