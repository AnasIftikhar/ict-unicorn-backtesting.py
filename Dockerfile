FROM python:3.12-slim

WORKDIR /app

# System deps needed to compile some Python packages (pandas_ta, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY unicorn.py .
COPY optimize.py .
COPY Binance-Vision.py .
COPY handler.py .

CMD ["python", "-u", "handler.py"]
