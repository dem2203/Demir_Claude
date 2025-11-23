# Multi-stage build for smaller image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Production image
FROM python:3.11-slim

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib.so.0 /usr/lib/
COPY --from=builder /usr/lib/libta_lib.so.0.0.0 /usr/lib/

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Some packages failed, installing core packages only..." && \
     pip install --no-cache-dir \
     python-dotenv==1.0.0 \
     aiohttp==3.9.1 \
     requests==2.31.0 \
     pandas==2.1.3 \
     numpy==1.24.3 \
     scipy==1.11.4 \
     ccxt==4.1.22 \
     websockets==12.0 \
     ta==0.11.0 \
     scikit-learn==1.3.2 \
     psycopg2-binary==2.9.9 \
     asyncpg==0.29.0 \
     fastapi==0.104.1 \
     uvicorn==0.24.0 \
     redis==5.0.1 \
     psutil==5.9.6)

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache

# Create non-root user
RUN useradd -m -u 1000 demirai && \
    chown -R demirai:demirai /app

# Copy application code
COPY --chown=demirai:demirai . .

# Switch to non-root user
USER demirai

# Environment variables (will be overridden by Railway)
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=UTC

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start command
CMD ["python", "main.py"]
