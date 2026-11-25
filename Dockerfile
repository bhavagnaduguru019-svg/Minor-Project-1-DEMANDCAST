# Multi-stage Dockerfile for DEMANDCAST

# --- Frontend Build Stage ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# --- Backend & Final Application Stage ---
FROM python:3.11-slim
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and ml code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY inventory/ ./inventory/
COPY utils/ ./utils/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

# Copy built frontend assets
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

ENV PYTHONPATH=/app/backend:/app

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
