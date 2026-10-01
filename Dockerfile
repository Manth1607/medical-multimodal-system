FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for ML packages (like easyocr, opencv, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for various services
EXPOSE 8000 8501 8080

# Default command (can be overridden in docker-compose.yml)
CMD ["bash"]
