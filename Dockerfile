FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --upgrade pip && \
    pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.6 && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install opencv-python-headless==4.8.1.78 && \
    pip install ultralytics numpy pillow pyyaml scikit-learn requests

# Copy application files
COPY yolo_loader_fix.py .
COPY basketball_referee.py .
COPY main.py .

# Create models directory
RUN mkdir -p models

# Download model at runtime via environment variable
ENV MODEL_URL=""

# Expose port
EXPOSE 10000

# Run the application
CMD ["python", "main.py"]