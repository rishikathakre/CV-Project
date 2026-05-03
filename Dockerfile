FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV and glib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY shared/ shared/
COPY src/ src/
COPY configs/ configs/
COPY yolov8n.pt .

# Data directory is mounted at runtime
VOLUME ["/app/data"]

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
