FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 5001

# Run with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "chatbot_backend:app"]
