FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend code
COPY chatbot_backend_claude_1.py /app/chatbot_backend.py

# 1) Install CPU-only PyTorch stack FIRST (no CUDA / nvidia deps)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    torchaudio==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest of your deps (without letting sentence-transformers pull torch again)
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    gunicorn \
    anthropic \
    scikit-learn \
    numpy \
    scipy

# 3) Install sentence-transformers WITHOUT dependencies (we already installed torch etc.)
RUN pip install --no-cache-dir \
    sentence-transformers==2.6.1 --no-deps

ENV PORT=5003
ENV ANTHROPIC_API_KEY=""

EXPOSE 5003

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} chatbot_backend:app"]
